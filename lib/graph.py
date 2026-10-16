from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088


def load_station_metadata(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"station_id", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if df[["lat", "lon"]].isna().any().any():
        raise ValueError("Station coordinates contain NaN values.")
    return df.reset_index(drop=True)


def build_haversine_knn(
    coords_deg: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    coords_deg = np.asarray(coords_deg, dtype=np.float64)
    if coords_deg.ndim != 2 or coords_deg.shape[1] != 2:
        raise ValueError("coords_deg must have shape [N, 2].")
    n = coords_deg.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < N, got k={k}, N={n}.")

    coords_rad = np.deg2rad(coords_deg)
    tree = BallTree(coords_rad, metric="haversine")
    distances_rad, indices = tree.query(coords_rad, k=k + 1)
    return indices[:, 1:].astype(np.int64), (distances_rad[:, 1:] * EARTH_RADIUS_KM).astype(np.float32)


def load_or_build_graph(
    metadata_path: str | Path,
    cache_path: str | Path,
    k: int,
    station_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    metadata = load_station_metadata(metadata_path)
    if station_indices is not None:
        station_indices = np.asarray(station_indices, dtype=np.int64)
        metadata = metadata.iloc[station_indices].reset_index(drop=True)

    cache_path = Path(cache_path)
    cache_is_usable = cache_path.exists() and station_indices is None
    if cache_is_usable:
        cached = np.load(cache_path)
        if int(cached["k"]) == k and len(cached["neighbor_indices"]) == len(metadata):
            return cached["neighbor_indices"], cached["distances_km"], metadata

    coords = metadata[["lat", "lon"]].to_numpy(np.float64)
    neighbor_indices, distances_km = build_haversine_knn(coords, k=k)
    if station_indices is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            neighbor_indices=neighbor_indices,
            distances_km=distances_km,
            k=np.asarray(k, dtype=np.int64),
        )
    return neighbor_indices, distances_km, metadata
