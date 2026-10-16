from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch import Tensor

from model.base import CARAMOEOutput, compute_availability_features
from model.components import (
    ExternalExpertLowRankAdapters,
    LaggedCrossPollutantAggregator,
    LastDailyBaselineRouter,
    MultiScaleSpatialAggregator,
    TwoLayerMaskedDecayGRUWithSnapshots,
)
from model.backbone import CARAMOEBackbone


class CARAMOE(CARAMOEBackbone):

    VARIANT_NAME = "CARAMOE"

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
        self_baseline: str = "last",
        trust_floor: float = 0.0,
        cross_mode: str = "current",
        lag_hours: Sequence[int] = (0, 1, 3, 6, 12, 23),
        adapter_rank: int = 0,
        spatial_scales: Sequence[int] = (10,),
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
            residual_self=True,
            temporal_layers=2,
        )
        self_baseline = self_baseline.lower()
        cross_mode = cross_mode.lower()
        scales = tuple(int(scale) for scale in spatial_scales)
        if self_baseline not in {"last", "last_daily"}:
            raise ValueError("self_baseline must be 'last' or 'last_daily'.")
        if not 0.0 <= trust_floor <= 1.0:
            raise ValueError("trust_floor must be between 0 and 1.")
        if cross_mode not in {"current", "lagged"}:
            raise ValueError("cross_mode must be 'current' or 'lagged'.")
        if adapter_rank < 0:
            raise ValueError("adapter_rank cannot be negative.")

        self.self_baseline = self_baseline
        self.trust_floor = float(trust_floor)
        self.cross_mode = cross_mode
        self.lag_hours = tuple(int(lag) for lag in lag_hours)
        self.adapter_rank = int(adapter_rank)
        self.spatial_scales = scales

        if self.self_baseline == "last_daily":
            self.baseline_router = LastDailyBaselineRouter(
                context_dim=hidden_dim + 5 + 2,
                hidden_dim=hidden_dim,
                output_horizon=output_horizon,
                dropout=dropout,
            )
        else:
            self.baseline_router = None

        if self.cross_mode == "lagged":
            self.temporal_encoder = TwoLayerMaskedDecayGRUWithSnapshots(
                hidden_dim=hidden_dim,
                station_dim=station_dim,
                pollutant_dim=pollutant_dim,
                time_feature_dim=time_feature_dim,
                time_dim=time_dim,
                num_pollutants=num_pollutants,
                dropout=dropout,
                lag_hours=self.lag_hours,
            )
            self.cross_aggregator = LaggedCrossPollutantAggregator(
                hidden_dim=hidden_dim,
                num_pollutants=num_pollutants,
                num_lags=len(self.lag_hours),
                chunk_size=spatial_chunk_size,
                dropout=dropout,
            )

        self.external_adapters = (
            ExternalExpertLowRankAdapters(hidden_dim, self.adapter_rank)
            if self.adapter_rank > 0
            else None
        )

        self.use_multiscale_spatial = self.spatial_scales != (10,)
        if self.use_multiscale_spatial:
            self.spatial_aggregator = MultiScaleSpatialAggregator(
                hidden_dim=hidden_dim,
                num_pollutants=num_pollutants,
                neighbor_indices=neighbor_indices,
                distances_km=distances_km,
                scales=self.spatial_scales,
                chunk_size=spatial_chunk_size,
                dropout=dropout,
            )

        self._original_fast_path = (
            self.self_baseline == "last"
            and self.trust_floor == 0.0
            and self.cross_mode == "current"
            and self.external_adapters is None
            and not self.use_multiscale_spatial
        )

    def experiment_metadata(self) -> dict[str, object]:
        return {
            "model_architecture": self.VARIANT_NAME,
            "self_baseline": self.self_baseline,
            "trust_floor": self.trust_floor,
            "cross_mode": self.cross_mode,
            "lag_hours": list(self.lag_hours),
            "adapter_rank": self.adapter_rank,
            "spatial_scales": list(self.spatial_scales),
            "residual_self": True,
            "temporal_layers": 2,
            "base_model_files_unchanged": [
                "model/base.py",
                "model/backbone.py",
            ],
        }

    def forward(self, x: Tensor, x_mask: Tensor, x_time: Tensor, y_time: Tensor) -> Dict[str, Tensor]:
        if self._original_fast_path:
            return super().forward(x, x_mask, x_time, y_time)
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

        lag_attention_summary = None
        if self.cross_mode == "lagged":
            temporal_state, snapshots, snapshot_freshness = self.temporal_encoder(
                x, x_mask, station, pollutant, x_time
            )
        else:
            temporal_state = self.temporal_encoder(x, x_mask, station, pollutant, x_time)
            snapshots = snapshot_freshness = None

        availability, self_reliability = compute_availability_features(x_mask)
        static = torch.cat(
            [
                station[None, :, None, :].expand(b, -1, self.num_pollutants, -1),
                pollutant[None, None, :, :].expand(b, self.num_stations, -1, -1),
            ],
            dim=-1,
        )
        query = self.query_encoder(
            torch.cat([temporal_state, static, availability, capability_b], dim=-1)
        )

        spatial_same_scale_weights = None
        spatial_cross_scale_weights = None
        if self.use_multiscale_spatial:
            spatial_same, spatial_rel, spatial_same_scale_weights = self.spatial_aggregator(
                query, temporal_state, self_reliability, availability
            )
        else:
            spatial_same, spatial_rel = self.spatial_aggregator(
                query, temporal_state, self_reliability
            )

        if self.cross_mode == "lagged":
            assert snapshots is not None and snapshot_freshness is not None
            cross_local, cross_rel, lag_attention_summary = self.cross_aggregator(
                query, snapshots, snapshot_freshness
            )
        else:
            cross_local, cross_rel = self.cross_aggregator(
                query, temporal_state, self_reliability
            )

        if self.use_multiscale_spatial:
            spatial_cross, spatial_cross_rel, spatial_cross_scale_weights = self.spatial_aggregator(
                query, cross_local, cross_rel, availability
            )
        else:
            spatial_cross, spatial_cross_rel = self.spatial_aggregator(
                query, cross_local, cross_rel
            )
        prior = self.prior_encoder(torch.cat([static, capability_b], dim=-1))
        prior_rel = torch.ones_like(self_reliability)

        expert_features_list = [temporal_state, spatial_same, cross_local, spatial_cross, prior]
        expert_features_list = [
            norm(feature) for norm, feature in zip(self.expert_norms, expert_features_list)
        ]
        adapter_norm_ratios = None
        if self.external_adapters is not None:
            adapted, adapter_norm_ratios = self.external_adapters(expert_features_list[1:])
            expert_features_list = [expert_features_list[0], *adapted]

        expert_features = torch.stack(expert_features_list, dim=-2)
        expert_reliability = torch.stack(
            [self_reliability, spatial_rel, cross_rel, spatial_cross_rel, prior_rel], dim=-1
        )

        if self.self_baseline == "last":
            expert_predictions = self._decode_experts(
                expert_features_list, x=x, x_mask=x_mask, y_time=y_time
            )
            baseline_weights = None
        else:
            assert self.baseline_router is not None
            baseline_context = torch.cat([temporal_state, availability, capability_b], dim=-1)
            baseline, baseline_weights, _ = self.baseline_router(
                baseline_context, x, x_mask, y_time.shape[1]
            )
            self_delta = self.self_delta_decoder(expert_features_list[0], y_time)
            self_prediction = baseline + self_delta
            external_predictions = [
                self.decoder(feature, y_time) for feature in expert_features_list[1:]
            ]
            expert_predictions = torch.stack(
                [self_prediction, *external_predictions], dim=-1
            )

        external_weights, external_gate_raw = self.router(
            query, expert_features, expert_reliability, availability, capability_b
        )
        trust_cap = self.trust_floor + (1.0 - self.trust_floor) * (
            1.0 - self_reliability[:, None].square()
        )
        external_gate = external_gate_raw * trust_cap
        self_prediction = expert_predictions[..., 0]
        external_predictions_tensor = expert_predictions[..., 1:]
        correction = torch.sum(
            external_weights * (external_predictions_tensor - self_prediction[..., None]), dim=-1
        )
        prediction = self_prediction + external_gate * correction

        result = CARAMOEOutput(
            prediction=prediction,
            expert_predictions=expert_predictions,
            expert_reliability=expert_reliability,
            external_weights=external_weights,
            external_gate=external_gate,
            self_reliability=self_reliability,
            availability_features=availability,
        ).as_dict()
        result["external_gate_raw"] = external_gate_raw
        if baseline_weights is not None:
            result["baseline_weights"] = baseline_weights
        if lag_attention_summary is not None:
            result["lag_attention_summary"] = lag_attention_summary
        if spatial_same_scale_weights is not None:
            result["spatial_same_scale_weights"] = spatial_same_scale_weights
        if spatial_cross_scale_weights is not None:
            result["spatial_cross_scale_weights"] = spatial_cross_scale_weights
        if adapter_norm_ratios is not None:
            result["adapter_norm_ratios"] = adapter_norm_ratios
        return result
