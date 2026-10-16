from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

POLLUTANTS = ("pm25", "pm10", "no2", "o3", "so2", "co")


def load_scaler(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if not {"mean", "std"}.issubset(df.columns):
        raise ValueError("Scaler CSV must contain mean and std columns.")
    mean = df["mean"].to_numpy(np.float32)
    std = df["std"].to_numpy(np.float32)
    if len(mean) != 6:
        raise ValueError(f"Expected six pollutant scalers, got {len(mean)}.")
    return mean, std


def _detect_h5_keys(h5: h5py.File) -> tuple[str, str]:
    value_key = "values" if "values" in h5 else "data" if "data" in h5 else None
    mask_key = "masks" if "masks" in h5 else "mask" if "mask" in h5 else None
    if value_key is None or mask_key is None:
        raise KeyError(f"Unsupported HDF5 layout. Found keys: {list(h5.keys())}")
    return value_key, mask_key


def cyclic_time_features(unix_hours: np.ndarray) -> np.ndarray:
    unix_hours = np.asarray(unix_hours, dtype=np.int64)
    seconds = unix_hours * 3600
    dt = pd.to_datetime(seconds, unit="s", utc=True)
    hour = dt.hour.to_numpy(np.float32)
    dow = dt.dayofweek.to_numpy(np.float32)
    doy = dt.dayofyear.to_numpy(np.float32)
    return np.stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.2425),
            np.cos(2 * np.pi * doy / 365.2425),
        ],
        axis=-1,
    ).astype(np.float32)


@dataclass(frozen=True)
class YearInfo:
    year: int
    path: Path
    length: int
    value_key: str
    mask_key: str
    start_unix_hour: int


def inspect_year_file(data_dir: str | Path, year: int) -> YearInfo:
    path = Path(data_dir) / f"aq_compact_{year}.h5"
    if not path.exists():
        path = Path(data_dir) / f"{year}.h5"
    if not path.exists():
        raise FileNotFoundError(f"Missing AirQualityBench file: {path}")
    with h5py.File(path, "r") as f:
        value_key, mask_key = _detect_h5_keys(f)
        if f[value_key].shape != f[mask_key].shape:
            raise ValueError(f"Value/mask shape mismatch in {path}.")
        length = int(f[value_key].shape[0])
        if f[value_key].shape[-1] != 6:
            raise ValueError(f"Expected six pollutants in {path}, got shape {f[value_key].shape}.")
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    start_unix_hour = int(start.timestamp() // 3600)
    return YearInfo(year, path, length, value_key, mask_key, start_unix_hour)


def compute_capability_profile(
    data_dir: str | Path,
    years: Iterable[int],
    output_path: str | Path,
    chunk_hours: int = 168,
) -> dict[str, np.ndarray]:
    counts: np.ndarray | None = None
    total_hours = 0
    for year in years:
        info = inspect_year_file(data_dir, int(year))
        with h5py.File(info.path, "r") as f:
            ds = f[info.mask_key]
            if counts is None:
                counts = np.zeros(ds.shape[1:], dtype=np.int64)
            for start in range(0, info.length, chunk_hours):
                stop = min(start + chunk_hours, info.length)
                counts += np.asarray(ds[start:stop], dtype=np.bool_).sum(axis=0, dtype=np.int64)
                total_hours += stop - start
    if counts is None or total_hours == 0:
        raise ValueError("No training masks were found.")
    coverage = counts.astype(np.float32) / float(total_hours)
    ever_observed = counts > 0
    result = {
        "coverage": coverage,
        "ever_observed": ever_observed.astype(np.float32),
        "counts": counts,
        "total_hours": np.asarray(total_hours, dtype=np.int64),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)
    return result


def load_or_compute_capability(
    data_dir: str | Path,
    train_years: Sequence[int],
    cache_path: str | Path,
) -> dict[str, np.ndarray]:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return compute_capability_profile(data_dir, train_years, cache_path)
    cached = np.load(cache_path)
    return {key: cached[key] for key in cached.files}


class AirQualityWindowDataset(Dataset):

    def __init__(
        self,
        data_dir: str | Path,
        years: Sequence[int],
        mean: np.ndarray,
        std: np.ndarray,
        input_window: int,
        output_window: int,
        step: int,
        station_indices: np.ndarray | None = None,
        cross_year_windows: bool = True,
    ) -> None:
        super().__init__()
        self.in_len = int(input_window)
        self.out_len = int(output_window)
        self.step = int(step)
        if self.in_len <= 0 or self.out_len <= 0 or self.step <= 0:
            raise ValueError("Window lengths and step must be positive.")

        self.years = [inspect_year_file(data_dir, int(y)) for y in years]
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 6)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 6)
        self.station_indices = None if station_indices is None else np.asarray(station_indices, dtype=np.int64)
        self.cross_year_windows = bool(cross_year_windows)
        self._handles: dict[int, h5py.File] = {}

        self.samples: list[tuple[int, int]] = []
        total = self.in_len + self.out_len
        self.year_offsets = [0]
        for info in self.years:
            self.year_offsets.append(self.year_offsets[-1] + info.length)

        if self.cross_year_windows and len(self.years) > 1:
            for left, right in zip(self.years, self.years[1:]):
                if left.start_unix_hour + left.length != right.start_unix_hour:
                    raise ValueError(
                        "cross_year_windows requires consecutive hourly year files; "
                        f"{left.year} and {right.year} are not contiguous."
                    )
            combined_length = self.year_offsets[-1]
            for start in range(0, combined_length - total + 1, self.step):
                self.samples.append((-1, start))
        else:
            for yi, info in enumerate(self.years):
                if info.length < total:
                    continue
                for start in range(0, info.length - total + 1, self.step):
                    self.samples.append((yi, start))
        if not self.samples:
            raise ValueError("No valid windows can be formed from the selected files.")

        with h5py.File(self.years[0].path, "r") as f:
            n = int(f[self.years[0].value_key].shape[1])
        self.num_stations = n if self.station_indices is None else len(self.station_indices)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __del__(self):
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def _get_handle(self, year_idx: int) -> h5py.File:
        handle = self._handles.get(year_idx)
        if handle is None:
            handle = h5py.File(self.years[year_idx].path, "r")
            self._handles[year_idx] = handle
        return handle

    def __len__(self) -> int:
        return len(self.samples)

    def _read_window(self, year_idx: int, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        info = self.years[year_idx]
        f = self._get_handle(year_idx)
        values = np.asarray(f[info.value_key][start:stop], dtype=np.float32)
        masks = np.asarray(f[info.mask_key][start:stop], dtype=np.bool_)
        unix_hours = info.start_unix_hour + np.arange(start, stop, dtype=np.int64)
        return values, masks, unix_hours

    def _read_global_window(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        value_parts: list[np.ndarray] = []
        mask_parts: list[np.ndarray] = []
        hour_parts: list[np.ndarray] = []
        cursor = start
        while cursor < stop:
            year_idx = bisect_right(self.year_offsets, cursor) - 1
            local_start = cursor - self.year_offsets[year_idx]
            local_stop = min(self.years[year_idx].length, stop - self.year_offsets[year_idx])
            values, masks, unix_hours = self._read_window(year_idx, local_start, local_stop)
            value_parts.append(values)
            mask_parts.append(masks)
            hour_parts.append(unix_hours)
            cursor += local_stop - local_start
        return (
            np.concatenate(value_parts, axis=0),
            np.concatenate(mask_parts, axis=0),
            np.concatenate(hour_parts, axis=0),
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        year_idx, start = self.samples[index]
        stop = start + self.in_len + self.out_len
        if year_idx == -1:
            values, masks, unix_hours = self._read_global_window(start, stop)
        else:
            values, masks, unix_hours = self._read_window(year_idx, start, stop)
        if self.station_indices is not None:
            values = values[:, self.station_indices, :]
            masks = masks[:, self.station_indices, :]

        x = values[: self.in_len]
        x_mask = masks[: self.in_len]
        y = values[self.in_len :]
        y_mask = masks[self.in_len :]

        x_norm = (x - self.mean) / (self.std + 1e-5)
        y_norm = (y - self.mean) / (self.std + 1e-5)
        x_norm[~x_mask] = 0.0
        y_norm[~y_mask] = 0.0

        time_features = cyclic_time_features(unix_hours)

        return {
            "x": torch.from_numpy(x_norm),
            "x_mask": torch.from_numpy(x_mask),
            "y": torch.from_numpy(y_norm),
            "y_mask": torch.from_numpy(y_mask),
            "x_time": torch.from_numpy(time_features[: self.in_len]),
            "y_time": torch.from_numpy(time_features[self.in_len :]),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }


def choose_station_subset(
    capability_coverage: np.ndarray,
    max_stations: int | None,
    explicit_path: str | None,
) -> np.ndarray | None:
    n = capability_coverage.shape[0]
    if explicit_path:
        indices = np.load(explicit_path).astype(np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("Invalid station subset indices.")
        return indices
    if max_stations is None or max_stations >= n:
        return None
    score = capability_coverage.sum(axis=1)
    return np.argsort(score)[-int(max_stations) :].astype(np.int64)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    pin_memory = num_workers > 0 and torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
