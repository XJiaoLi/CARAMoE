from pathlib import Path
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _decode_strings(values):
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _daily_time_features(date_strings):
    dates = np.asarray(date_strings, dtype="datetime64[D]")
    day_number = dates.astype(np.int64)
    day_of_week = ((day_number + 3) % 7).astype(np.float32)
    year_start = dates.astype("datetime64[Y]").astype("datetime64[D]")
    day_of_year = ((dates - year_start).astype(np.int64) + 1).astype(np.float32)
    zeros = np.zeros(dates.shape[0], dtype=np.float32)
    ones = np.ones(dates.shape[0], dtype=np.float32)
    return np.stack(
        [
            zeros,
            ones,
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
            np.sin(2 * np.pi * day_of_year / 365.2425),
            np.cos(2 * np.pi * day_of_year / 365.2425),
        ],
        axis=-1,
    ).astype(np.float32)


class GlobalAQDataset(Dataset):
    def __init__(
        self,
        data_dir,
        years,
        scaler_dict,
        adj_path,
        window_in=30,
        window_out=1,
        step=1,
        expected_params=None,
        expected_frequency="daily",
        split_name="dataset",
        return_context=False,
        mask_overlay_dir=None,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.years = [int(year) for year in years]
        self.window_in = int(window_in)
        self.window_out = int(window_out)
        self.step = int(step)
        self.split_name = str(split_name)
        self.expected_frequency = str(expected_frequency)
        self.return_context = bool(return_context)
        self.mask_overlay_dir = (
            Path(mask_overlay_dir).resolve() if mask_overlay_dir else None
        )

        if not self.years:
            raise ValueError(f"{self.split_name}: years cannot be empty")
        if self.years != sorted(self.years) or len(self.years) != len(set(self.years)):
            raise ValueError(f"{self.split_name}: years must be unique and sorted")
        if self.window_in <= 0 or self.window_out <= 0 or self.step <= 0:
            raise ValueError("window_in, window_out, and step must be positive")

        adj_path = Path(adj_path).resolve()
        if not adj_path.exists():
            raise FileNotFoundError(f"Adjacency matrix not found: {adj_path}")
        with adj_path.open("rb") as handle:
            adj_data = pickle.load(handle)
        if not {"adj_mx", "sensor_ids"}.issubset(adj_data):
            raise ValueError("Adjacency file must contain adj_mx and sensor_ids")

        adj_array = np.asarray(adj_data["adj_mx"], dtype=np.float32)
        self.station_ids = [str(station_id) for station_id in adj_data["sensor_ids"]]
        if adj_array.ndim != 2 or adj_array.shape[0] != adj_array.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got {adj_array.shape}")
        if adj_array.shape[0] != len(self.station_ids):
            raise ValueError("Adjacency size does not match sensor_ids length")
        if not np.isfinite(adj_array).all():
            raise ValueError("Adjacency matrix contains non-finite values")
        self.adj_mx = torch.from_numpy(adj_array).float()

        means = np.asarray(scaler_dict["mean"], dtype=np.float32)
        stds = np.asarray(scaler_dict["std"], dtype=np.float32)
        if means.ndim != 1 or stds.ndim != 1 or means.shape != stds.shape:
            raise ValueError("Scaler mean/std must be one-dimensional arrays with equal length")
        if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
            raise ValueError("Scaler values must be finite and every std must be positive")
        self.mean = torch.from_numpy(means).view(1, 1, -1).float()
        self.std = torch.from_numpy(stds).view(1, 1, -1).float()

        all_values = []
        all_masks = []
        all_input_masks = []
        all_dates = []
        params = None
        previous_last_date = None
        expected_params = list(expected_params) if expected_params is not None else None

        for year in self.years:
            file_path = self.data_dir / f"aq_compact_{year}.h5"
            if not file_path.exists():
                file_path = self.data_dir / f"{year}.h5"
            if not file_path.exists():
                raise FileNotFoundError(f"{self.split_name}: missing HDF5 file {file_path}")

            with h5py.File(file_path, "r") as handle:
                required = {"values", "masks", "station_ids", "params", "dates"}
                missing = sorted(required.difference(handle.keys()))
                if missing:
                    raise ValueError(f"{file_path.name}: missing HDF5 datasets {missing}")

                frequency = handle.attrs.get("frequency", "")
                if isinstance(frequency, bytes):
                    frequency = frequency.decode("utf-8")
                if str(frequency) != self.expected_frequency:
                    raise ValueError(
                        f"{file_path.name}: frequency={frequency!r}, "
                        f"expected {self.expected_frequency!r}"
                    )

                values = np.asarray(handle["values"][:], dtype=np.float32)
                masks = np.asarray(handle["masks"][:], dtype=np.bool_)
                file_station_ids = _decode_strings(handle["station_ids"][:])
                file_params = _decode_strings(handle["params"][:])
                date_strings = _decode_strings(handle["dates"][:])

            if values.ndim != 3 or values.shape != masks.shape:
                raise ValueError(
                    f"{file_path.name}: values/masks must share (T,N,C), got "
                    f"{values.shape} and {masks.shape}"
                )
            if values.shape[1] != len(self.station_ids):
                raise ValueError(
                    f"{file_path.name}: node count {values.shape[1]} != "
                    f"adjacency node count {len(self.station_ids)}"
                )
            if values.shape[2] != means.size:
                raise ValueError(
                    f"{file_path.name}: feature count {values.shape[2]} != "
                    f"scaler feature count {means.size}"
                )
            if file_station_ids != self.station_ids:
                raise ValueError(f"{file_path.name}: station order differs from adjacency sensor_ids")
            if expected_params is not None and file_params != expected_params:
                raise ValueError(
                    f"{file_path.name}: parameter order {file_params} != {expected_params}"
                )
            if params is None:
                params = file_params
            elif file_params != params:
                raise ValueError(f"{file_path.name}: parameter order differs across yearly files")
            if masks.shape[0] != len(date_strings):
                raise ValueError(f"{file_path.name}: dates length does not match time dimension")
            if not np.isfinite(values[masks]).all():
                raise ValueError(f"{file_path.name}: valid-mask positions contain non-finite values")

            input_masks = masks
            if self.mask_overlay_dir is not None:
                overlay_path = self.mask_overlay_dir / f"mask_overlay_{year}.npz"
                if not overlay_path.exists():
                    overlay_path = self.mask_overlay_dir / f"{year}.npz"
                if not overlay_path.exists():
                    raise FileNotFoundError(
                        f"{self.split_name}: missing mask overlay {overlay_path}"
                    )
                with np.load(overlay_path, allow_pickle=False) as overlay:
                    required_overlay = {
                        "synthetic_keep_mask", "effective_mask", "drop_reason",
                        "dates", "station_ids", "params", "year",
                    }
                    missing_overlay = sorted(required_overlay.difference(overlay.files))
                    if missing_overlay:
                        raise ValueError(
                            f"{overlay_path.name}: missing arrays {missing_overlay}"
                        )
                    overlay_year = int(np.asarray(overlay["year"]).item())
                    synthetic_keep = np.asarray(
                        overlay["synthetic_keep_mask"], dtype=np.bool_
                    )
                    effective_mask = np.asarray(overlay["effective_mask"], dtype=np.bool_)
                    drop_reason = np.asarray(overlay["drop_reason"], dtype=np.uint8)
                    overlay_dates = _decode_strings(overlay["dates"])
                    overlay_station_ids = _decode_strings(overlay["station_ids"])
                    overlay_params = _decode_strings(overlay["params"])

                if overlay_year != year:
                    raise ValueError(
                        f"{overlay_path.name}: stored year {overlay_year} != requested {year}"
                    )
                if synthetic_keep.shape != masks.shape or effective_mask.shape != masks.shape:
                    raise ValueError(
                        f"{overlay_path.name}: mask shape must be {masks.shape}, got "
                        f"{synthetic_keep.shape} and {effective_mask.shape}"
                    )
                if drop_reason.shape != masks.shape:
                    raise ValueError(
                        f"{overlay_path.name}: drop_reason shape {drop_reason.shape} != {masks.shape}"
                    )
                if overlay_dates != date_strings:
                    raise ValueError(f"{overlay_path.name}: dates differ from source HDF5")
                if overlay_station_ids != file_station_ids:
                    raise ValueError(f"{overlay_path.name}: station order differs from source HDF5")
                if overlay_params != file_params:
                    raise ValueError(f"{overlay_path.name}: parameter order differs from source HDF5")
                if not np.array_equal(effective_mask, masks & synthetic_keep):
                    raise ValueError(
                        f"{overlay_path.name}: effective_mask != source_mask & synthetic_keep_mask"
                    )
                if np.any(drop_reason > 4):
                    raise ValueError(f"{overlay_path.name}: drop_reason contains values outside 0..4")
                expected_natural = ~masks
                expected_synthetic = masks & ~synthetic_keep
                if not np.array_equal(drop_reason == 1, expected_natural):
                    raise ValueError(
                        f"{overlay_path.name}: reason 1 must exactly mark natural missing cells"
                    )
                if not np.array_equal(
                    np.isin(drop_reason, (2, 3, 4)), expected_synthetic
                ):
                    raise ValueError(
                        f"{overlay_path.name}: reasons 2..4 must exactly mark synthetic drops"
                    )
                input_masks = effective_mask

            dates = np.asarray(date_strings, dtype="datetime64[D]")
            expected_start = np.datetime64(f"{year}-01-01", "D")
            expected_end = np.datetime64(f"{year}-12-31", "D")
            if dates[0] != expected_start or dates[-1] != expected_end:
                raise ValueError(
                    f"{file_path.name}: expected full calendar year "
                    f"{expected_start}..{expected_end}, got {dates[0]}..{dates[-1]}"
                )
            if dates.size > 1 and not np.all(np.diff(dates).astype(int) == 1):
                raise ValueError(f"{file_path.name}: dates are not daily and continuous")
            if previous_last_date is not None and dates[0] != previous_last_date + np.timedelta64(1, "D"):
                raise ValueError(f"{file_path.name}: yearly files are not temporally continuous")
            previous_last_date = dates[-1]

            all_values.append(values)
            all_masks.append(masks)
            all_input_masks.append(input_masks)
            all_dates.extend(date_strings)

        raw_values = np.concatenate(all_values, axis=0)
        raw_masks = np.concatenate(all_masks, axis=0)
        raw_input_masks = np.concatenate(all_input_masks, axis=0)
        total_window = self.window_in + self.window_out
        if raw_values.shape[0] < total_window:
            raise ValueError(
                f"{self.split_name}: {raw_values.shape[0]} time steps are insufficient "
                f"for window {self.window_in}+{self.window_out}"
            )

        self.params = params or []
        self.dates = all_dates
        self.time_features = torch.from_numpy(_daily_time_features(all_dates)).float()
        self.num_nodes = raw_values.shape[1]
        self.num_features = raw_values.shape[2]
        self.values = torch.from_numpy(raw_values).float()
        self.masks = torch.from_numpy(raw_masks).bool()
        self.input_masks = torch.from_numpy(raw_input_masks).bool()
        self.values = (self.values - self.mean) / (self.std + 1e-5)
        self.values[~self.masks] = 0.0

        natural_coverage = float(self.masks.float().mean())
        input_coverage = float(self.input_masks.float().mean())
        print(
            f"Dataset ready | split={self.split_name} | years={self.years} | "
            f"frequency={self.expected_frequency} | stations={self.num_nodes} | "
            f"features={self.num_features} | samples={len(self)} | "
            f"natural_coverage={natural_coverage:.4f} | input_coverage={input_coverage:.4f}"
        )

    def get_adj(self):
        return self.adj_mx

    def get_capability_profile(self):
        coverage = self.input_masks.float().mean(dim=0).numpy().astype(np.float32)
        ever_observed = self.input_masks.any(dim=0).float().numpy().astype(np.float32)
        return coverage, ever_observed

    def __len__(self):
        total_window = self.window_in + self.window_out
        return (self.values.shape[0] - total_window) // self.step + 1

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        start_idx = idx * self.step
        x_end = start_idx + self.window_in
        y_end = x_end + self.window_out

        x = self.values[start_idx:x_end, :, :].clone()
        y = self.values[x_end:y_end, :, :]
        x_mask = self.input_masks[start_idx:x_end, :, :]
        y_mask = self.masks[x_end:y_end, :, :]
        x[~x_mask] = 0.0
        if self.return_context:
            return {
                "x": x,
                "x_mask": x_mask,
                "y": y,
                "y_mask": y_mask,
                "x_time": self.time_features[start_idx:x_end, :],
                "y_time": self.time_features[x_end:y_end, :],
            }
        return x, y, y_mask
