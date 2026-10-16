from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model.base import MLP, _masked_softmax
from model.backbone import TwoLayerMaskedDecayGRU, last_observed_value


def last_daily_candidates(
    x: Tensor,
    mask: Tensor,
    horizon: int,
    daily_period: int = 24,
) -> tuple[Tensor, Tensor]:

    if x.shape != mask.shape or x.ndim != 4:
        raise ValueError("x and mask must have the same [B,T,N,P] shape.")
    if horizon <= 0 or daily_period <= 0:
        raise ValueError("horizon and daily_period must be positive.")

    mask = mask.bool()
    b, t_len, n, p = x.shape
    last, last_valid = last_observed_value(x, mask)
    last = last[:, None].expand(b, horizon, n, p)
    last_valid = last_valid[:, None].expand(b, horizon, n, p)

    daily_indices = torch.arange(horizon, device=x.device) + t_len - daily_period
    in_range = (daily_indices >= 0) & (daily_indices < t_len)
    safe_indices = daily_indices.clamp(0, max(t_len - 1, 0))
    daily = x.index_select(1, safe_indices)
    daily_valid = mask.index_select(1, safe_indices) & in_range[None, :, None, None]
    daily = torch.where(daily_valid, daily, torch.zeros_like(daily))

    candidates = torch.stack([last, daily], dim=-1)
    valid = torch.stack([last_valid, daily_valid], dim=-1)
    return candidates, valid


class LastDailyBaselineRouter(nn.Module):

    def __init__(self, context_dim: int, hidden_dim: int, output_horizon: int, dropout: float):
        super().__init__()
        self.output_horizon = output_horizon
        self.router = MLP(context_dim, output_horizon * 2, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        context: Tensor,
        x: Tensor,
        mask: Tensor,
        horizon: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if horizon > self.output_horizon:
            raise ValueError(
                f"Baseline router was built for {self.output_horizon} steps, received {horizon}."
            )
        b, n, p, _ = context.shape
        logits = self.router(context).view(b, n, p, self.output_horizon, 2)
        logits = logits[..., :horizon, :].permute(0, 3, 1, 2, 4)
        candidates, valid = last_daily_candidates(x, mask, horizon)
        weights = _masked_softmax(logits, valid, dim=-1)
        baseline = (weights * candidates).sum(dim=-1)
        return baseline, weights, valid


class TwoLayerMaskedDecayGRUWithSnapshots(TwoLayerMaskedDecayGRU):

    def __init__(self, *args, lag_hours: Sequence[int], **kwargs):
        super().__init__(*args, **kwargs)
        lags = tuple(int(lag) for lag in lag_hours)
        if not lags or min(lags) < 0 or len(set(lags)) != len(lags):
            raise ValueError("lag_hours must contain unique non-negative integers.")
        self.lag_hours = lags

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        station_features: Tensor,
        pollutant_features: Tensor,
        time_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, t_len, n, p = x.shape
        if max(self.lag_hours) >= t_len:
            raise ValueError(
                f"Largest lag {max(self.lag_hours)} is invalid for input length {t_len}."
            )
        mask = mask.bool()
        static = torch.cat(
            [
                station_features[None, :, None, :].expand(b, n, p, -1),
                pollutant_features[None, None, :, :].expand(b, n, p, -1),
            ],
            dim=-1,
        )
        states = [initializer(static) for initializer in self.init_states]
        gamma = torch.exp(-F.softplus(self.raw_decay))[:, None, None, :, :]
        encoded_time = self.time_encoder(time_features)
        snapshot_times = {t_len - 1 - lag: index for index, lag in enumerate(self.lag_hours)}
        snapshots: list[Tensor | None] = [None] * len(self.lag_hours)
        freshness: list[Tensor | None] = [None] * len(self.lag_hours)
        gap = torch.zeros((b, n, p), dtype=x.dtype, device=x.device)
        seen = torch.zeros((b, n, p), dtype=torch.bool, device=x.device)

        for tau in range(t_len):
            observed_scalar = mask[:, tau]
            observed = observed_scalar[..., None]
            first_input = torch.cat(
                [
                    x[:, tau, :, :, None],
                    static,
                    encoded_time[:, tau, None, None, :].expand(b, n, p, -1),
                ],
                dim=-1,
            )
            first_decayed = states[0] * gamma[0]
            first_candidate = self.gru_layers[0](
                first_input.reshape(b * n * p, -1),
                first_decayed.reshape(b * n * p, -1),
            ).reshape(b, n, p, self.hidden_dim)
            states[0] = torch.where(observed, self.dropout(first_candidate), first_decayed)

            second_decayed = states[1] * gamma[1]
            second_candidate = self.gru_layers[1](
                states[0].reshape(b * n * p, -1),
                second_decayed.reshape(b * n * p, -1),
            ).reshape(b, n, p, self.hidden_dim)
            states[1] = torch.where(observed, self.dropout(second_candidate), second_decayed)

            seen = seen | observed_scalar
            gap = torch.where(observed_scalar, torch.zeros_like(gap), gap + 1.0)
            if tau in snapshot_times:
                index = snapshot_times[tau]
                snapshots[index] = states[1]
                freshness[index] = torch.exp(-gap / 6.0) * seen.float()

        if any(value is None for value in snapshots) or any(value is None for value in freshness):
            raise RuntimeError("Failed to collect every requested lag snapshot.")
        snapshot_tensor = torch.stack([value for value in snapshots if value is not None], dim=-2)
        freshness_tensor = torch.stack([value for value in freshness if value is not None], dim=-1)
        return states[1], snapshot_tensor, freshness_tensor


class LaggedCrossPollutantAggregator(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_pollutants: int,
        num_lags: int,
        chunk_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_pollutants = num_pollutants
        self.num_lags = num_lags
        self.chunk_size = chunk_size
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.relation_lag_bias = nn.Parameter(
            torch.zeros(num_pollutants, num_pollutants, num_lags)
        )

    def forward(
        self,
        query: Tensor,
        snapshots: Tensor,
        source_freshness: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, n, p, lags, d = snapshots.shape
        if p != self.num_pollutants or lags != self.num_lags:
            raise ValueError("Snapshot pollutant/lag dimensions do not match the aggregator.")
        outputs, coverages = [], []
        attention_sum = snapshots.new_zeros((p, p, lags), dtype=torch.float32)
        attention_count = 0
        offdiag = ~torch.eye(p, dtype=torch.bool, device=snapshots.device)[None, None, :, :, None]
        bias = self.relation_lag_bias[None, None]
        for start in range(0, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            q = self.q_proj(query[:, start:stop])
            src = snapshots[:, start:stop]
            k = self.k_proj(src)
            v = self.v_proj(src)
            scores = torch.einsum("bctd,bcsld->bctsl", q, k) / math.sqrt(d)
            src_rel = source_freshness[:, start:stop]
            valid = (src_rel[:, :, None] > 1e-6) & offdiag
            scores = scores + bias + torch.log(src_rel[:, :, None].clamp_min(1e-6))
            flat_scores = scores.flatten(-2)
            flat_valid = valid.flatten(-2)
            flat_weights = _masked_softmax(flat_scores, flat_valid, dim=-1)
            weights = flat_weights.view(b, stop - start, p, p, lags)
            out = torch.einsum("bctsl,bcsld->bctd", weights, v)
            coverage = torch.einsum("bctsl,bcsl->bct", weights, src_rel)
            outputs.append(self.out_proj(out))
            coverages.append(coverage)
            attention_sum += weights.detach().float().sum(dim=(0, 1))
            attention_count += b * (stop - start)
        summary = attention_sum / max(attention_count, 1)
        return (
            torch.cat(outputs, dim=1),
            torch.cat(coverages, dim=1).clamp(0.0, 1.0),
            summary,
        )


class ExternalExpertLowRankAdapters(nn.Module):

    def __init__(self, hidden_dim: int, rank: int, num_external_experts: int = 4):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.down = nn.ModuleList(
            [nn.Linear(hidden_dim, rank, bias=False) for _ in range(num_external_experts)]
        )
        self.up = nn.ModuleList(
            [nn.Linear(rank, hidden_dim, bias=False) for _ in range(num_external_experts)]
        )
        self.alpha = nn.Parameter(torch.ones(num_external_experts))
        for layer in self.up:
            nn.init.zeros_(layer.weight)

    def forward(self, features: Sequence[Tensor]) -> tuple[list[Tensor], Tensor]:
        if len(features) != len(self.down):
            raise ValueError(f"Expected {len(self.down)} external features, received {len(features)}.")
        adapted, ratios = [], []
        for index, feature in enumerate(features):
            delta = self.up[index](F.gelu(self.down[index](feature))) * self.alpha[index]
            adapted.append(feature + delta)
            ratios.append(
                delta.detach().float().norm() / feature.detach().float().norm().clamp_min(1e-8)
            )
        return adapted, torch.stack(ratios)


class MultiScaleSpatialAggregator(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_pollutants: int,
        neighbor_indices: np.ndarray,
        distances_km: np.ndarray,
        scales: Sequence[int],
        chunk_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        scales = tuple(int(scale) for scale in scales)
        if not scales or min(scales) <= 0 or tuple(sorted(set(scales))) != scales:
            raise ValueError("scales must be unique positive integers in ascending order.")
        if max(scales) > neighbor_indices.shape[1]:
            raise ValueError("The graph cache has fewer neighbors than the largest scale.")
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.scales = scales
        self.register_buffer(
            "neighbor_indices", torch.as_tensor(neighbor_indices, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "distances_km", torch.as_tensor(distances_km, dtype=torch.float32), persistent=True
        )
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.raw_distance_scale = nn.Parameter(torch.full((num_pollutants,), -2.0))
        self.scale_router = MLP(hidden_dim + 5, len(scales), hidden_dim=32, dropout=dropout)

    def forward(
        self,
        query: Tensor,
        source_states: Tensor,
        source_reliability: Tensor,
        availability: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, n, p, d = source_states.shape
        outputs, coverages, route_summaries = [], [], []
        distance_scale = F.softplus(self.raw_distance_scale)[None, None, None, :]
        for start in range(0, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            max_k = max(self.scales)
            idx = self.neighbor_indices[start:stop, :max_k]
            dist = self.distances_km[start:stop, :max_k]
            q = self.q_proj(query[:, start:stop])
            src = source_states[:, idx]
            src_rel = source_reliability[:, idx]
            k = self.k_proj(src)
            v = self.v_proj(src)
            base_scores = torch.einsum("bcpd,bckpd->bckp", q, k) / math.sqrt(d)
            base_scores = base_scores - torch.log1p(dist)[None, :, :, None] * distance_scale
            base_scores = base_scores + torch.log(src_rel.clamp_min(1e-6))

            scale_outputs, scale_coverages = [], []
            for scale in self.scales:
                valid = src_rel[:, :, :scale] > 1e-6
                weights = _masked_softmax(base_scores[:, :, :scale], valid, dim=2)
                scale_outputs.append(
                    torch.einsum("bckp,bckpd->bcpd", weights, v[:, :, :scale])
                )
                scale_coverages.append(
                    torch.einsum("bckp,bckp->bcp", weights, src_rel[:, :, :scale])
                )
            stacked_outputs = torch.stack(scale_outputs, dim=-2)
            stacked_coverages = torch.stack(scale_coverages, dim=-1)
            scale_logits = self.scale_router(
                torch.cat([query[:, start:stop], availability[:, start:stop]], dim=-1)
            )
            scale_weights = torch.softmax(scale_logits, dim=-1)
            combined = (stacked_outputs * scale_weights[..., None]).sum(dim=-2)
            coverage = (stacked_coverages * scale_weights).sum(dim=-1)
            outputs.append(self.out_proj(combined))
            coverages.append(coverage)
            route_summaries.append(scale_weights.detach().float().sum(dim=(0, 1)))
        summary = torch.stack(route_summaries).sum(dim=0) / max(b * n, 1)
        return (
            torch.cat(outputs, dim=1),
            torch.cat(coverages, dim=1).clamp(0.0, 1.0),
            summary,
        )
