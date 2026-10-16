from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from model import CARAMOE
from model.ablation import (
    CARAMOEAblation,
    LEAVE_ONE_OUT_GROUPS,
)


EARTH_RADIUS_KM = 6371.0088


DEFAULT_F1_CONFIG = {
    "hidden_dim": 48,
    "station_dim": 24,
    "pollutant_dim": 16,
    "time_dim": 16,
    "router_dim": 32,
    "spatial_chunk_size": 256,
    "dropout": 0.1,
    "station_id_dropout": 0.3,
    "use_station_id": True,
    "decoder_layers": 2,
    "input_profile": "full",
    "self_baseline": "last",
    "trust_floor": 0.1,
    "cross_mode": "lagged",
    "lag_steps": [0, 1, 3, 6, 12, 23],
    "adapter_rank": 8,
    "spatial_scales": [5, 10],
    "knn_k": 10,
    "ablation": "full",
    "leave_one_out": "none",
}


CONTROLLED_ABLATIONS = {
    "uniform_routing": "c2_uniform_external",
    "static_routing": "c3_static_routing",
    "capability_only": "c6_capability_only",
    "neither_availability_nor_capability": "c7_neither_availability_nor_capability",
}
ABLATION_CHOICES = {"full", "self_only", *CONTROLLED_ABLATIONS}


def _build_haversine_knn(coords_deg, k):
    coords = np.asarray(coords_deg, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coordinates [N,2], got {coords.shape}")
    if not np.isfinite(coords).all():
        raise ValueError("Station coordinates contain non-finite values")
    station_count = coords.shape[0]
    if not 1 <= k < station_count:
        raise ValueError(f"knn_k must satisfy 1 <= k < {station_count}, got {k}")

    radians = np.deg2rad(coords)
    lat = radians[:, 0]
    lon = radians[:, 1]
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    haversine = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    distances = 2.0 * EARTH_RADIUS_KM * np.arcsin(
        np.sqrt(np.clip(haversine, 0.0, 1.0))
    )
    np.fill_diagonal(distances, np.inf)
    neighbor_indices = np.argsort(distances, axis=1, kind="stable")[:, :k]
    row_indices = np.arange(station_count)[:, None]
    neighbor_distances = distances[row_indices, neighbor_indices]
    return neighbor_indices.astype(np.int64), neighbor_distances.astype(np.float32)


def _load_ordered_coordinates(metadata_path, expected_station_ids):
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Station metadata not found: {path}")
    metadata = pd.read_csv(path, encoding="utf-8-sig")
    required = {"station_id", "lat", "lon"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Station metadata is missing columns: {missing}")
    metadata_ids = metadata["station_id"].astype(str).tolist()
    expected_ids = [str(station_id) for station_id in expected_station_ids]
    if metadata_ids != expected_ids:
        raise ValueError("Station metadata order does not match the HDF5/adjacency node order")
    return metadata[["lat", "lon"]].to_numpy(dtype=np.float32)


class CARAMOEChinaAir(nn.Module):

    requires_context = True
    supports_full_f1_objective = True

    def __init__(
        self,
        coords_deg,
        neighbor_indices,
        distances_km,
        capability_coverage,
        capability_ever,
        output_horizon,
        input_window,
        model_config=None,
    ):
        super().__init__()
        config = dict(DEFAULT_F1_CONFIG)
        config.update(dict(model_config or {}))
        self.input_window = int(input_window)
        self.output_horizon = int(output_horizon)
        lag_steps = tuple(int(value) for value in config["lag_steps"])
        spatial_scales = tuple(int(value) for value in config["spatial_scales"])
        ablation = str(config["ablation"]).strip().lower().replace("-", "_")
        leave_one_out = (
            str(config["leave_one_out"]).strip().lower().replace("-", "_")
        )
        if ablation not in ABLATION_CHOICES:
            raise ValueError(
                f"Unknown CARAMOE ablation {ablation!r}; "
                f"choose one of {sorted(ABLATION_CHOICES)}"
            )
        if leave_one_out not in LEAVE_ONE_OUT_GROUPS:
            raise ValueError(
                f"Unknown CARAMOE leave-one-out group {leave_one_out!r}; "
                f"choose one of {sorted(LEAVE_ONE_OUT_GROUPS)}"
            )
        if leave_one_out != "none" and ablation != "full":
            raise ValueError("Expert leave-one-out requires ablation='full'")
        if not spatial_scales or min(spatial_scales) <= 0:
            raise ValueError("F1 spatial_scales must contain positive integers")
        if max(lag_steps) >= self.input_window:
            raise ValueError(
                f"F1 maximum lag step {max(lag_steps)} must be smaller than input window "
                f"{self.input_window}"
            )
        if max(spatial_scales) > neighbor_indices.shape[1]:
            raise ValueError("F1 graph has fewer neighbors than its largest spatial scale")
        if int(config["knn_k"]) != int(neighbor_indices.shape[1]):
            raise ValueError("F1 knn_k must equal the supplied graph neighbor count")

        core_kwargs = dict(
            coords_deg=np.asarray(coords_deg, dtype=np.float32),
            neighbor_indices=np.asarray(neighbor_indices, dtype=np.int64),
            distances_km=np.asarray(distances_km, dtype=np.float32),
            capability_coverage=np.asarray(capability_coverage, dtype=np.float32),
            capability_ever=np.asarray(capability_ever, dtype=np.float32),
            output_horizon=self.output_horizon,
            hidden_dim=int(config["hidden_dim"]),
            station_dim=int(config["station_dim"]),
            pollutant_dim=int(config["pollutant_dim"]),
            time_dim=int(config["time_dim"]),
            router_dim=int(config["router_dim"]),
            num_pollutants=6,
            spatial_chunk_size=int(config["spatial_chunk_size"]),
            dropout=float(config["dropout"]),
            station_id_dropout=float(config["station_id_dropout"]),
            use_station_id=bool(config["use_station_id"]),
            decoder_layers=int(config["decoder_layers"]),
            time_feature_dim=6,
            input_profile=str(config["input_profile"]),
            self_baseline=str(config["self_baseline"]),
            trust_floor=float(config["trust_floor"]),
            cross_mode=str(config["cross_mode"]),
            lag_hours=lag_steps,
            adapter_rank=int(config["adapter_rank"]),
            spatial_scales=spatial_scales,
        )
        controlled_ablation = CONTROLLED_ABLATIONS.get(ablation)
        if controlled_ablation is None and leave_one_out == "none":
            self.core = CARAMOE(**core_kwargs)
        else:
            self.core = CARAMOEAblation(
                **core_kwargs,
                ablation=controlled_ablation or "full",
                leave_one_out=leave_one_out,
            )
        self.model_config = config
        self.ablation = ablation
        self.leave_one_out = leave_one_out

    @classmethod
    def from_training_dataset(cls, dataset, metadata_path, output_horizon, model_config=None):
        config = dict(DEFAULT_F1_CONFIG)
        config.update(dict(model_config or {}))
        coords = _load_ordered_coordinates(metadata_path, dataset.station_ids)
        neighbor_indices, distances_km = _build_haversine_knn(coords, int(config["knn_k"]))
        capability_coverage, capability_ever = dataset.get_capability_profile()
        return cls(
            coords_deg=coords,
            neighbor_indices=neighbor_indices,
            distances_km=distances_km,
            capability_coverage=capability_coverage,
            capability_ever=capability_ever,
            output_horizon=output_horizon,
            input_window=dataset.window_in,
            model_config=config,
        )

    def _validate_inputs(self, x, x_mask, x_time, y_time):
        if x_mask is None or x_time is None or y_time is None:
            raise ValueError("CARAMOE requires x_mask, x_time, and y_time")
        if x_mask.shape != x.shape:
            raise ValueError(f"x_mask shape {x_mask.shape} does not match x shape {x.shape}")
        if x.shape[1] != self.input_window:
            raise ValueError(f"Expected input window {self.input_window}, got {x.shape[1]}")
        if x_time.shape != (x.shape[0], x.shape[1], 6):
            raise ValueError(f"Unexpected x_time shape: {x_time.shape}")
        if y_time.shape != (x.shape[0], self.output_horizon, 6):
            raise ValueError(f"Unexpected y_time shape: {y_time.shape}")

    def forward_full(self, x, x_mask=None, x_time=None, y_time=None):
        self._validate_inputs(x, x_mask, x_time, y_time)
        output = self.core(x, x_mask.bool(), x_time, y_time)
        if self.ablation == "self_only":
            output = dict(output)
            output["prediction"] = output["expert_predictions"][..., 0]
            output["external_weights"] = torch.zeros_like(output["external_weights"])
            output["external_gate"] = torch.zeros_like(output["external_gate"])
            if "external_gate_raw" in output:
                output["external_gate_raw"] = torch.zeros_like(output["external_gate_raw"])
            reliability = output["expert_reliability"].clone()
            reliability[..., 1:] = 0
            output["expert_reliability"] = reliability
            output["active_external_expert_mask"] = torch.zeros(
                4,
                dtype=output["prediction"].dtype,
                device=output["prediction"].device,
            )
        expected = (x.shape[0], self.output_horizon, x.shape[2], x.shape[3])
        if output["prediction"].shape != expected:
            raise ValueError(
                f"Unexpected CARAMOE prediction shape: {output['prediction'].shape}"
            )
        return output

    def forward(self, x, x_mask=None, x_time=None, y_time=None):
        return self.forward_full(x, x_mask=x_mask, x_time=x_time, y_time=y_time)[
            "prediction"
        ]

    def experiment_metadata(self):
        metadata = dict(self.core.experiment_metadata())
        router_mode = metadata.get(
            "router_mode",
            "self_only" if self.ablation == "self_only" else "adaptive_factorized",
        )
        if self.leave_one_out != "none":
            objective_name = (
                f"F1-full LOO remove-{self.leave_one_out} aqb_mix objective"
            )
        elif self.ablation == "full":
            objective_name = "F1-full CARAMOE aqb_mix composite objective"
        elif self.ablation == "self_only":
            objective_name = "C1 self-only aqb_mix objective"
        else:
            objective_name = f"Controlled {self.ablation} aqb_mix ablation objective"
        metadata.update(
            {
                "benchmark_name": "caramoe",
                "source_class": type(self.core).__name__,
                "frequency": "daily",
                "lag_unit": "daily_step",
                "knn_k": int(self.model_config["knn_k"]),
                "ablation": self.ablation,
                "canonical_ablation": CONTROLLED_ABLATIONS.get(
                    self.ablation, self.ablation
                ),
                "leave_one_out": self.leave_one_out,
                "router_mode": router_mode,
                "training_objective": objective_name,
                "training_objective_components": [
                    "0.7 * physical-unit global masked MAE",
                    "0.3 * pollutant-balanced normalized masked MAE",
                    "0.15 * reliability-weighted expert auxiliary MAE",
                    "0.10 * high-support safe-routing hinge",
                    "0.05 * route consistency every 4 batches",
                ],
            }
        )
        return metadata
