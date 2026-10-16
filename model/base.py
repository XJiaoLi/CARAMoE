from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class CARAMOEOutput:
    prediction: Tensor
    expert_predictions: Tensor
    expert_reliability: Tensor
    external_weights: Tensor
    external_gate: Tensor
    self_reliability: Tensor
    availability_features: Tensor

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            "prediction": self.prediction,
            "expert_predictions": self.expert_predictions,
            "expert_reliability": self.expert_reliability,
            "external_weights": self.external_weights,
            "external_gate": self.external_gate,
            "self_reliability": self.self_reliability,
            "availability_features": self.availability_features,
        }


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CoordinateEncoder(nn.Module):

    def __init__(self, coords_deg: np.ndarray, out_dim: int, dropout: float = 0.0):
        super().__init__()
        coords = torch.as_tensor(coords_deg, dtype=torch.float32)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords_deg must have shape [N, 2] in latitude/longitude order.")
        lat = torch.deg2rad(coords[:, 0])
        lon = torch.deg2rad(coords[:, 1])
        xyz = torch.stack(
            [torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)], dim=-1
        )
        feats = [xyz, torch.sin(lat[:, None]), torch.cos(lat[:, None]), torch.sin(lon[:, None]), torch.cos(lon[:, None])]
        for freq in (2.0, 4.0, 8.0):
            feats.extend(
                [
                    torch.sin(freq * lat[:, None]),
                    torch.cos(freq * lat[:, None]),
                    torch.sin(freq * lon[:, None]),
                    torch.cos(freq * lon[:, None]),
                ]
            )
        fixed = torch.cat(feats, dim=-1)
        self.register_buffer("fixed_features", fixed, persistent=True)
        self.encoder = MLP(fixed.shape[-1], out_dim, hidden_dim=2 * out_dim, dropout=dropout)

    def forward(self) -> Tensor:
        return self.encoder(self.fixed_features)


class MaskedDecayGRU(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        station_dim: int,
        pollutant_dim: int,
        time_feature_dim: int,
        time_dim: int,
        num_pollutants: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_encoder = MLP(time_feature_dim, time_dim, hidden_dim=2 * time_dim, dropout=dropout)
        input_dim = 1 + station_dim + pollutant_dim + time_dim
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.init_state = MLP(station_dim + pollutant_dim, hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.raw_decay = nn.Parameter(torch.full((num_pollutants, hidden_dim), -2.0))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        station_features: Tensor,
        pollutant_features: Tensor,
        time_features: Tensor,
    ) -> Tensor:
        b, t_len, n, p = x.shape
        static = torch.cat(
            [
                station_features[None, :, None, :].expand(b, n, p, -1),
                pollutant_features[None, None, :, :].expand(b, n, p, -1),
            ],
            dim=-1,
        )
        h = self.init_state(static)
        gamma = torch.exp(-F.softplus(self.raw_decay))[None, None, :, :]
        encoded_time = self.time_encoder(time_features)

        for tau in range(t_len):
            h_decayed = h * gamma
            inp = torch.cat(
                [
                    x[:, tau, :, :, None],
                    static,
                    encoded_time[:, tau, None, None, :].expand(b, n, p, -1),
                ],
                dim=-1,
            )
            candidate = self.gru(inp.reshape(b * n * p, -1), h_decayed.reshape(b * n * p, -1))
            candidate = self.dropout(candidate).reshape(b, n, p, self.hidden_dim)
            observed = mask[:, tau, :, :, None]
            h = torch.where(observed, candidate, h_decayed)
        return h


def compute_availability_features(mask: Tensor) -> tuple[Tensor, Tensor]:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    b, t_len, n, p = mask.shape
    current = mask[:, -1].float()
    ratio_all = mask.float().mean(dim=1)
    recent_len = min(6, t_len)
    ratio_recent = mask[:, -recent_len:].float().mean(dim=1)
    reversed_mask = mask.flip(1)
    has_any = reversed_mask.any(dim=1)
    trailing_gap = reversed_mask.float().argmax(dim=1).float()
    trailing_gap = torch.where(has_any, trailing_gap, torch.full_like(trailing_gap, float(t_len)))
    gap_norm = trailing_gap / max(float(t_len), 1.0)
    freshness = torch.exp(-trailing_gap / 6.0)
    features = torch.stack([current, ratio_all, ratio_recent, gap_norm, freshness], dim=-1)
    reliability = (0.35 * current + 0.25 * ratio_recent + 0.20 * ratio_all + 0.20 * freshness).clamp(0.0, 1.0)
    reliability = reliability * has_any.float()
    return features, reliability


def _masked_softmax(scores: Tensor, valid: Tensor, dim: int) -> Tensor:
    scores = scores.masked_fill(~valid, -1e4)
    weights = torch.softmax(scores, dim=dim)
    weights = weights * valid.float()
    denom = weights.sum(dim=dim, keepdim=True)
    return weights / denom.clamp_min(1e-8)


class CrossPollutantAggregator(nn.Module):
    def __init__(self, hidden_dim: int, num_pollutants: int, chunk_size: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_pollutants = num_pollutants
        self.chunk_size = chunk_size
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.relation_bias = nn.Parameter(torch.zeros(num_pollutants, num_pollutants))

    def forward(self, query: Tensor, states: Tensor, source_reliability: Tensor) -> tuple[Tensor, Tensor]:
        b, n, p, d = states.shape
        outputs, coverages = [], []
        offdiag = ~torch.eye(p, dtype=torch.bool, device=states.device)[None, None, :, :]
        relation_bias = self.relation_bias[None, None, :, :]
        for start in range(0, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            q = self.q_proj(query[:, start:stop])
            src = states[:, start:stop]
            k = self.k_proj(src)
            v = self.v_proj(src)
            scores = torch.einsum("bctd,bcsd->bcts", q, k) / math.sqrt(d)
            src_rel = source_reliability[:, start:stop]
            valid = (src_rel[:, :, None, :] > 1e-6) & offdiag
            scores = scores + relation_bias + torch.log(src_rel[:, :, None, :].clamp_min(1e-6))
            weights = _masked_softmax(scores, valid, dim=-1)
            out = torch.einsum("bcts,bcsd->bctd", weights, v)
            coverage = torch.einsum("bcts,bcs->bct", weights, src_rel)
            outputs.append(self.out_proj(out))
            coverages.append(coverage)
        return torch.cat(outputs, dim=1), torch.cat(coverages, dim=1).clamp(0.0, 1.0)


class ChunkedSpatialAggregator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_pollutants: int,
        neighbor_indices: np.ndarray,
        distances_km: np.ndarray,
        chunk_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.register_buffer("neighbor_indices", torch.as_tensor(neighbor_indices, dtype=torch.long), persistent=True)
        self.register_buffer("distances_km", torch.as_tensor(distances_km, dtype=torch.float32), persistent=True)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.raw_distance_scale = nn.Parameter(torch.full((num_pollutants,), -2.0))

    def forward(self, query: Tensor, source_states: Tensor, source_reliability: Tensor) -> tuple[Tensor, Tensor]:
        b, n, p, d = source_states.shape
        outputs, coverages = [], []
        distance_scale = F.softplus(self.raw_distance_scale)[None, None, None, :]
        for start in range(0, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            idx = self.neighbor_indices[start:stop]
            dist = self.distances_km[start:stop]
            q = self.q_proj(query[:, start:stop])
            src = source_states[:, idx, :, :]
            src_rel = source_reliability[:, idx, :]
            k = self.k_proj(src)
            v = self.v_proj(src)
            scores = torch.einsum("bcpd,bckpd->bckp", q, k) / math.sqrt(d)
            scores = scores - torch.log1p(dist)[None, :, :, None] * distance_scale
            scores = scores + torch.log(src_rel.clamp_min(1e-6))
            valid = src_rel > 1e-6
            weights = _masked_softmax(scores, valid, dim=2)
            out = torch.einsum("bckp,bckpd->bcpd", weights, v)
            coverage = torch.einsum("bckp,bckp->bcp", weights, src_rel)
            outputs.append(self.out_proj(out))
            coverages.append(coverage)
        return torch.cat(outputs, dim=1), torch.cat(coverages, dim=1).clamp(0.0, 1.0)


class SharedHorizonDecoder(nn.Module):
    def __init__(self, hidden_dim: int, time_feature_dim: int, output_horizon: int, layers: int, dropout: float):
        super().__init__()
        self.output_horizon = output_horizon
        self.time_proj = MLP(time_feature_dim, hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.horizon_embedding = nn.Embedding(output_horizon, hidden_dim)
        blocks: list[nn.Module] = []
        for _ in range(max(1, layers - 1)):
            blocks.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        blocks.append(nn.Linear(hidden_dim, 1))
        self.decoder = nn.Sequential(*blocks)

    def forward(self, feature: Tensor, future_time: Tensor) -> Tensor:
        b, n, p, d = feature.shape
        h = future_time.shape[1]
        if h > self.output_horizon:
            raise ValueError(f"Decoder was built for {self.output_horizon} steps, received {h}.")
        time = self.time_proj(future_time)[:, :, None, None, :]
        horizon_ids = torch.arange(h, device=feature.device)
        horizon = self.horizon_embedding(horizon_ids)[None, :, None, None, :]
        fused = feature[:, None, :, :, :] + time + horizon
        return self.decoder(fused).squeeze(-1)


class AvailabilityRouter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        route_dim: int,
        num_experts: int,
        output_horizon: int,
        availability_dim: int,
        capability_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.output_horizon = output_horizon
        self.q_proj = nn.Linear(hidden_dim, route_dim, bias=False)
        self.e_proj = nn.Linear(hidden_dim, route_dim, bias=False)
        context_in = hidden_dim + availability_dim + capability_dim + num_experts
        low_rank = min(8, route_dim)
        self.route_context = MLP(context_in, low_rank, hidden_dim=route_dim, dropout=dropout)
        self.route_horizon = nn.Parameter(torch.randn(output_horizon, low_rank, num_experts) * 0.02)
        self.horizon_bias = nn.Parameter(torch.zeros(output_horizon, num_experts))

        self.beta_context = MLP(context_in, low_rank, hidden_dim=route_dim, dropout=dropout)
        self.beta_horizon = nn.Parameter(torch.randn(output_horizon, low_rank) * 0.02)
        self.beta_bias = nn.Parameter(torch.full((output_horizon,), -3.0))
        self.raw_lack_weights = nn.Parameter(torch.tensor([[0.7, 0.5, 0.3, 0.6, 0.2]]).repeat(output_horizon, 1))

    def forward(
        self,
        query: Tensor,
        expert_features: Tensor,
        expert_reliability: Tensor,
        availability: Tensor,
        capability: Tensor,
    ) -> tuple[Tensor, Tensor]:
        q = self.q_proj(query)
        e = self.e_proj(expert_features)
        base_scores = torch.einsum("bnpr,bnper->bnpe", q, e) / math.sqrt(q.shape[-1])
        base_scores = base_scores + torch.log(expert_reliability.clamp_min(1e-6))

        context_input = torch.cat([query, availability, capability, expert_reliability], dim=-1)
        route_ctx = self.route_context(context_input)
        horizon_logits = torch.einsum("bnpr,hre->bhnpe", route_ctx, self.route_horizon)
        logits = base_scores[:, None] + horizon_logits + self.horizon_bias[None, :, None, None, :]

        external_logits = logits[..., 1:]
        external_valid = expert_reliability[:, None, :, :, 1:] > 1e-6
        external_weights = _masked_softmax(external_logits, external_valid, dim=-1)

        current = availability[..., 0]
        ratio_all = availability[..., 1]
        ratio_recent = availability[..., 2]
        gap_norm = availability[..., 3]
        capability_rate = capability[..., 0]
        lack = torch.stack(
            [1.0 - current, 1.0 - ratio_recent, 1.0 - ratio_all, gap_norm, 1.0 - capability_rate], dim=-1
        )
        lack_weights = F.softplus(self.raw_lack_weights)
        monotone_term = torch.einsum("bnpf,hf->bhnp", lack, lack_weights)
        beta_ctx = self.beta_context(context_input)
        beta_context_term = torch.einsum("bnpr,hr->bhnp", beta_ctx, self.beta_horizon)
        beta = torch.sigmoid(self.beta_bias[None, :, None, None] + monotone_term + beta_context_term)
        return external_weights, beta


class CARAMOEBase(nn.Module):

    EXPERT_NAMES: Sequence[str] = ("self", "spatial_same", "cross_local", "spatial_cross", "prior")

    def __init__(
        self,
        coords_deg: np.ndarray,
        neighbor_indices: np.ndarray,
        distances_km: np.ndarray,
        capability_coverage: np.ndarray,
        capability_ever: np.ndarray,
        output_horizon: int,
        hidden_dim: int = 48,
        station_dim: int = 24,
        pollutant_dim: int = 16,
        time_dim: int = 16,
        router_dim: int = 32,
        num_pollutants: int = 6,
        spatial_chunk_size: int = 256,
        dropout: float = 0.1,
        station_id_dropout: float = 0.3,
        use_station_id: bool = True,
        decoder_layers: int = 2,
        time_feature_dim: int = 6,
        input_profile: str = "full",
    ) -> None:
        super().__init__()
        n = len(coords_deg)
        if capability_coverage.shape != (n, num_pollutants):
            raise ValueError("capability_coverage must have shape [N, P].")
        if capability_ever.shape != (n, num_pollutants):
            raise ValueError("capability_ever must have shape [N, P].")
        self.num_stations = n
        self.num_pollutants = num_pollutants
        self.output_horizon = output_horizon
        self.station_id_dropout = station_id_dropout
        self.input_profile = input_profile.lower()
        if self.input_profile not in {"strict", "full"}:
            raise ValueError(f"input_profile must be strict or full, got {input_profile!r}.")
        self.use_extra_features = self.input_profile == "full"
        self.use_station_id = bool(use_station_id and self.use_extra_features)

        self.coordinate_encoder = (
            CoordinateEncoder(coords_deg, station_dim, dropout=dropout)
            if self.use_extra_features
            else None
        )
        self.register_buffer(
            "strict_station_features", torch.zeros(n, station_dim, dtype=torch.float32), persistent=False
        )
        self.station_embedding = nn.Embedding(n, station_dim) if self.use_station_id else None
        self.pollutant_embedding = nn.Embedding(num_pollutants, pollutant_dim)
        capability_coverage_tensor = torch.as_tensor(capability_coverage, dtype=torch.float32)
        capability_ever_tensor = torch.as_tensor(capability_ever, dtype=torch.float32)
        if not self.use_extra_features:
            capability_coverage_tensor = torch.zeros_like(capability_coverage_tensor)
            capability_ever_tensor = torch.zeros_like(capability_ever_tensor)
        self.register_buffer("capability_coverage", capability_coverage_tensor)
        self.register_buffer("capability_ever", capability_ever_tensor)

        self.temporal_encoder = MaskedDecayGRU(
            hidden_dim=hidden_dim,
            station_dim=station_dim,
            pollutant_dim=pollutant_dim,
            time_feature_dim=time_feature_dim,
            time_dim=time_dim,
            num_pollutants=num_pollutants,
            dropout=dropout,
        )
        query_in = hidden_dim + station_dim + pollutant_dim + 5 + 2
        self.query_encoder = MLP(query_in, hidden_dim, hidden_dim=2 * hidden_dim, dropout=dropout)
        self.spatial_aggregator = ChunkedSpatialAggregator(
            hidden_dim,
            num_pollutants,
            neighbor_indices,
            distances_km,
            spatial_chunk_size,
            dropout,
        )
        self.cross_aggregator = CrossPollutantAggregator(hidden_dim, num_pollutants, spatial_chunk_size, dropout)
        self.prior_encoder = MLP(station_dim + pollutant_dim + 2, hidden_dim, hidden_dim=2 * hidden_dim, dropout=dropout)
        self.expert_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.EXPERT_NAMES])
        self.decoder = SharedHorizonDecoder(
            hidden_dim, time_feature_dim, output_horizon, layers=decoder_layers, dropout=dropout
        )
        self.router = AvailabilityRouter(
            hidden_dim=hidden_dim,
            route_dim=router_dim,
            num_experts=len(self.EXPERT_NAMES),
            output_horizon=output_horizon,
            availability_dim=5,
            capability_dim=2,
            dropout=dropout,
        )

    def station_features(self) -> Tensor:
        station = (
            self.coordinate_encoder()
            if self.coordinate_encoder is not None
            else self.strict_station_features
        )
        if self.station_embedding is not None:
            ids = torch.arange(self.num_stations, device=station.device)
            id_feature = self.station_embedding(ids)
            if self.training and self.station_id_dropout > 0:
                keep = torch.rand(self.num_stations, 1, device=station.device) >= self.station_id_dropout
                id_feature = id_feature * keep
            station = station + id_feature
        return station

    def forward(self, x: Tensor, x_mask: Tensor, x_time: Tensor, y_time: Tensor) -> Dict[str, Tensor]:
        if x.ndim != 4 or x.shape[2:] != (self.num_stations, self.num_pollutants):
            raise ValueError(
                f"Expected x [B,T,{self.num_stations},{self.num_pollutants}], got {tuple(x.shape)}."
            )
        x_mask = x_mask.bool()
        if not self.use_extra_features:
            x_time = torch.zeros_like(x_time)
            y_time = torch.zeros_like(y_time)
        b = x.shape[0]
        station = self.station_features()
        pollutant = self.pollutant_embedding(torch.arange(self.num_pollutants, device=x.device))
        capability = torch.stack([self.capability_coverage, self.capability_ever], dim=-1)
        capability_b = capability[None].expand(b, -1, -1, -1)

        temporal_state = self.temporal_encoder(x, x_mask, station, pollutant, x_time)
        availability, self_reliability = compute_availability_features(x_mask)
        static = torch.cat(
            [
                station[None, :, None, :].expand(b, -1, self.num_pollutants, -1),
                pollutant[None, None, :, :].expand(b, self.num_stations, -1, -1),
            ],
            dim=-1,
        )
        query = self.query_encoder(torch.cat([temporal_state, static, availability, capability_b], dim=-1))

        spatial_same, spatial_rel = self.spatial_aggregator(query, temporal_state, self_reliability)
        cross_local, cross_rel = self.cross_aggregator(query, temporal_state, self_reliability)
        spatial_cross, spatial_cross_rel = self.spatial_aggregator(query, cross_local, cross_rel)
        prior = self.prior_encoder(torch.cat([static, capability_b], dim=-1))
        prior_rel = torch.ones_like(self_reliability)

        expert_features_list = [temporal_state, spatial_same, cross_local, spatial_cross, prior]
        expert_features_list = [norm(feat) for norm, feat in zip(self.expert_norms, expert_features_list)]
        expert_features = torch.stack(expert_features_list, dim=-2)
        expert_reliability = torch.stack(
            [self_reliability, spatial_rel, cross_rel, spatial_cross_rel, prior_rel], dim=-1
        )

        expert_predictions = torch.stack(
            [self.decoder(feat, y_time) for feat in expert_features_list], dim=-1
        )
        external_weights, external_gate_raw = self.router(
            query, expert_features, expert_reliability, availability, capability_b
        )
        trust_cap = 1.0 - self_reliability[:, None].square()
        external_gate = external_gate_raw * trust_cap
        self_prediction = expert_predictions[..., 0]
        external_predictions = expert_predictions[..., 1:]
        correction = torch.sum(
            external_weights * (external_predictions - self_prediction[..., None]), dim=-1
        )
        prediction = self_prediction + external_gate * correction

        return CARAMOEOutput(
            prediction=prediction,
            expert_predictions=expert_predictions,
            expert_reliability=expert_reliability,
            external_weights=external_weights,
            external_gate=external_gate,
            self_reliability=self_reliability,
            availability_features=availability,
        ).as_dict()
