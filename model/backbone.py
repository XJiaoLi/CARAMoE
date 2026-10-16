from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model.base import CARAMOEBase, CARAMOEOutput, MLP, SharedHorizonDecoder, compute_availability_features


def last_observed_value(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:

    if x.shape != mask.shape or x.ndim != 4:
        raise ValueError("x and mask must have the same [B,T,N,P] shape.")
    mask = mask.bool()
    time_length = x.shape[1]
    reversed_mask = mask.flip(1)
    has_observation = reversed_mask.any(dim=1)
    trailing_offset = reversed_mask.to(torch.int64).argmax(dim=1)
    last_index = (time_length - 1 - trailing_offset).clamp_min(0)
    gathered = torch.gather(x, dim=1, index=last_index[:, None]).squeeze(1)
    last_value = torch.where(has_observation, gathered, torch.zeros_like(gathered))
    return last_value, has_observation


class TwoLayerMaskedDecayGRU(nn.Module):

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
        first_input_dim = 1 + station_dim + pollutant_dim + time_dim
        self.gru_layers = nn.ModuleList(
            [
                nn.GRUCell(first_input_dim, hidden_dim),
                nn.GRUCell(hidden_dim, hidden_dim),
            ]
        )
        static_dim = station_dim + pollutant_dim
        self.init_states = nn.ModuleList(
            [
                MLP(static_dim, hidden_dim, hidden_dim=hidden_dim, dropout=dropout),
                MLP(static_dim, hidden_dim, hidden_dim=hidden_dim, dropout=dropout),
            ]
        )
        self.raw_decay = nn.Parameter(torch.full((2, num_pollutants, hidden_dim), -2.0))
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
        states = [initializer(static) for initializer in self.init_states]
        gamma = torch.exp(-F.softplus(self.raw_decay))[:, None, None, :, :]
        encoded_time = self.time_encoder(time_features)

        for tau in range(t_len):
            observed = mask[:, tau, :, :, None].bool()
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
            first_candidate = self.dropout(first_candidate)
            states[0] = torch.where(observed, first_candidate, first_decayed)

            second_decayed = states[1] * gamma[1]
            second_candidate = self.gru_layers[1](
                states[0].reshape(b * n * p, -1),
                second_decayed.reshape(b * n * p, -1),
            ).reshape(b, n, p, self.hidden_dim)
            second_candidate = self.dropout(second_candidate)
            states[1] = torch.where(observed, second_candidate, second_decayed)

        return states[1]


class CARAMOEBackbone(CARAMOEBase):

    VARIANT_NAME = "variant_base"

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
        *,
        residual_self: bool,
        temporal_layers: int,
    ) -> None:
        super().__init__(
            coords_deg=coords_deg,
            neighbor_indices=neighbor_indices,
            distances_km=distances_km,
            capability_coverage=capability_coverage,
            capability_ever=capability_ever,
            output_horizon=output_horizon,
            hidden_dim=hidden_dim,
            station_dim=station_dim,
            pollutant_dim=pollutant_dim,
            time_dim=time_dim,
            router_dim=router_dim,
            num_pollutants=num_pollutants,
            spatial_chunk_size=spatial_chunk_size,
            dropout=dropout,
            station_id_dropout=station_id_dropout,
            use_station_id=use_station_id,
            decoder_layers=decoder_layers,
            time_feature_dim=time_feature_dim,
            input_profile=input_profile,
        )
        if temporal_layers not in {1, 2}:
            raise ValueError("temporal_layers must be 1 or 2.")
        self.residual_self = bool(residual_self)
        self.temporal_layers = int(temporal_layers)

        if temporal_layers == 2:
            self.temporal_encoder = TwoLayerMaskedDecayGRU(
                hidden_dim=hidden_dim,
                station_dim=station_dim,
                pollutant_dim=pollutant_dim,
                time_feature_dim=time_feature_dim,
                time_dim=time_dim,
                num_pollutants=num_pollutants,
                dropout=dropout,
            )
        if self.residual_self:
            self.self_delta_decoder = SharedHorizonDecoder(
                hidden_dim,
                time_feature_dim,
                output_horizon,
                layers=decoder_layers,
                dropout=dropout,
            )

    def _decode_experts(
        self,
        expert_features: Sequence[Tensor],
        x: Tensor,
        x_mask: Tensor,
        y_time: Tensor,
    ) -> Tensor:
        if not self.residual_self:
            return torch.stack([self.decoder(feat, y_time) for feat in expert_features], dim=-1)

        last_value, _ = last_observed_value(x, x_mask)
        self_delta = self.self_delta_decoder(expert_features[0], y_time)
        self_prediction = last_value[:, None] + self_delta
        external_predictions = [self.decoder(feat, y_time) for feat in expert_features[1:]]
        return torch.stack([self_prediction, *external_predictions], dim=-1)

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
        expert_features_list = [
            norm(feat) for norm, feat in zip(self.expert_norms, expert_features_list)
        ]
        expert_features = torch.stack(expert_features_list, dim=-2)
        expert_reliability = torch.stack(
            [self_reliability, spatial_rel, cross_rel, spatial_cross_rel, prior_rel], dim=-1
        )
        expert_predictions = self._decode_experts(
            expert_features_list,
            x=x,
            x_mask=x_mask,
            y_time=y_time,
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
