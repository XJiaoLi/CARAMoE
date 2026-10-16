from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def cyclic_time_features(unix_hours: np.ndarray) -> np.ndarray:
    unix_hours = np.asarray(unix_hours, dtype=np.int64)
    dt = pd.to_datetime(unix_hours * 3600, unit="s", utc=True)
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


class KDDWindowDataset(Dataset):

    def __init__(
        self,
        h5_path: str | Path,
        manifest_path: str | Path,
        split: str,
        mean: np.ndarray,
        std: np.ndarray,
        input_window: int,
        output_window: int,
        step: int,
        station_indices: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.h5_path = Path(h5_path)
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.in_len = int(input_window)
        self.out_len = int(output_window)
        self.step = int(step)
        if self.in_len <= 0 or self.out_len <= 0 or self.step <= 0:
            raise ValueError("Window lengths and step must be positive.")
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 6)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 6)
        self.station_indices = (
            None if station_indices is None else np.asarray(station_indices, dtype=np.int64)
        )
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        split_info = self.manifest["splits"][split]
        self.split_start = int(split_info["start_index_inclusive"])
        self.split_end = int(split_info["end_index_exclusive"])
        first_target = max(self.split_start, self.in_len)
        last_target = self.split_end - self.out_len
        self.target_starts = (
            np.arange(first_target, last_target + 1, self.step, dtype=np.int64)
            if first_target <= last_target
            else np.empty(0, dtype=np.int64)
        )
        if self.target_starts.size == 0:
            raise ValueError(
                f"No {split} windows for input={self.in_len}, output={self.out_len}, step={self.step}."
            )
        self._handle: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as archive:
            if archive["values"].shape != archive["masks"].shape:
                raise ValueError("Value/mask shape mismatch.")
            if archive["values"].shape[-1] != 6:
                raise ValueError("KDD protocol tensors must contain six pollutant channels.")
            station_count = int(archive["values"].shape[1])
            self.station_ids = json.loads(archive.attrs["station_ids_json"])
        if self.station_indices is not None:
            if np.any(self.station_indices < 0) or np.any(self.station_indices >= station_count):
                raise ValueError("Invalid station indices.")
            self.station_ids = [self.station_ids[index] for index in self.station_indices]
        self.num_stations = len(self.station_ids)

    def __len__(self) -> int:
        return int(self.target_starts.size)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def _archive(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target_start = int(self.target_starts[index])
        window_start = target_start - self.in_len
        window_end = target_start + self.out_len
        archive = self._archive()
        values = np.asarray(archive["values"][window_start:window_end], dtype=np.float32)
        masks = np.asarray(archive["masks"][window_start:window_end], dtype=np.bool_)
        unix_hours = np.asarray(
            archive["timestamps_unix_hours"][window_start:window_end], dtype=np.int64
        )
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
            "target_start_index": torch.tensor(target_start, dtype=torch.long),
            "target_unix_hours": torch.from_numpy(unix_hours[self.in_len :].copy()),
        }
