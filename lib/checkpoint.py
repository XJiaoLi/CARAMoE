from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from torch import nn

from lib.config import ExperimentConfig
from lib.graph import build_haversine_knn, load_station_metadata
from model.ablation import CARAMOEAblation
from model.caramoe import CARAMOE


def _load_capability_readonly(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Capability cache is required for a read-only analysis: {path}"
        )
    with np.load(path) as archive:
        required = {"coverage", "ever_observed"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"Capability cache lacks {sorted(missing)}: {path}")
        return {name: np.asarray(archive[name]) for name in archive.files}

def _load_graph_readonly(
    config: ExperimentConfig,
    graph_k: int,
    subset: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metadata = load_station_metadata(config.data.metadata_path)
    if subset is not None:
        metadata = metadata.iloc[subset].reset_index(drop=True)
    coords = metadata[["lat", "lon"]].to_numpy(np.float32)
    station_ids = metadata["station_id"].astype(str).to_numpy()
    cache_path = Path(config.data.graph_cache_path)
    if subset is None and cache_path.exists():
        with np.load(cache_path) as archive:
            if (
                {"k", "neighbor_indices", "distances_km"}.issubset(archive.files)
                and int(archive["k"]) == graph_k
                and len(archive["neighbor_indices"]) == len(metadata)
            ):
                return (
                    np.asarray(archive["neighbor_indices"], dtype=np.int64),
                    np.asarray(archive["distances_km"], dtype=np.float32),
                    coords,
                    station_ids,
                )
    neighbors, distances = build_haversine_knn(coords, k=graph_k)
    return neighbors, distances, coords, station_ids

def model_from_metadata(
    config: ExperimentConfig,
    metadata: Mapping[str, object],
    *,
    coords_deg: np.ndarray,
    neighbor_indices: np.ndarray,
    distances_km: np.ndarray,
    capability_coverage: np.ndarray,
    capability_ever: np.ndarray,
) -> nn.Module:
    common = dict(
        coords_deg=coords_deg,
        neighbor_indices=neighbor_indices,
        distances_km=distances_km,
        capability_coverage=capability_coverage,
        capability_ever=capability_ever,
        output_horizon=config.data.output_window,
        hidden_dim=config.model.hidden_dim,
        station_dim=config.model.station_dim,
        pollutant_dim=config.model.pollutant_dim,
        time_dim=config.model.time_dim,
        router_dim=config.model.router_dim,
        num_pollutants=config.model.num_pollutants,
        spatial_chunk_size=config.model.spatial_chunk_size,
        dropout=config.model.dropout,
        station_id_dropout=config.model.station_id_dropout,
        use_station_id=config.model.use_station_id,
        decoder_layers=config.model.decoder_layers,
        input_profile=config.model.input_profile,
        self_baseline=str(metadata.get("self_baseline", "last")),
        trust_floor=float(metadata.get("requested_trust_floor", metadata.get("trust_floor", 0.0))),
        cross_mode=str(metadata.get("cross_mode", "current")),
        lag_hours=tuple(int(value) for value in metadata.get("lag_hours", (0, 1, 3, 6, 12, 23))),
        adapter_rank=int(metadata.get("adapter_rank", 0)),
        spatial_scales=tuple(int(value) for value in metadata.get("spatial_scales", (10,))),
    )
    family = str(metadata.get("experiment_family", ""))
    architecture = str(metadata.get("model_architecture", ""))
    if family == "CARAMOE_ablation" or architecture == CARAMOEAblation.VARIANT_NAME:
        return CARAMOEAblation(
            **common,
            ablation=str(metadata.get("ablation", "full")),
            leave_one_out=str(metadata.get("leave_one_out", "none")),
            excluded_experts=tuple(str(value) for value in metadata.get("excluded_experts", ())),
            freshness_mode=str(metadata.get("freshness_mode", "decay")),
        )
    if architecture in {"", CARAMOE.VARIANT_NAME}:
        return CARAMOE(**common)
    raise ValueError(
        f"Unsupported checkpoint architecture={architecture!r}, experiment_family={family!r}."
    )

