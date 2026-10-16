from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


POLLUTANTS = ("pm25", "pm10", "no2", "o3", "so2", "co")
POLLUTANT_ALIASES = {
    "PM2.5": "pm25",
    "PM25": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "O3": "o3",
    "SO2": "so2",
    "CO": "co",
}
CITY_KEYS = {"Beijing": "beijing", "London": "london"}
SPLIT_NAMES = ("train", "val", "test")
SPLIT_WEIGHTS = (3, 1, 1)
HORIZONS = (6, 12, 18, 24)
INPUT_WINDOW = 24


@dataclass(frozen=True)
class TSFSeries:
    series_name: str
    city: str
    station: str
    pollutant: str
    start_raw: str
    start_unix_hour: int
    values: np.ndarray
    mask: np.ndarray


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def _parse_clock_hour(raw: str) -> int:
    parsed = datetime.strptime(raw, "%Y-%m-%d %H-%M-%S")
    return calendar.timegm(parsed.timetuple()) // 3600


def _clock_iso(unix_hour: int) -> str:
    return datetime.fromtimestamp(int(unix_hour) * 3600, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


def read_tsf(path: str | Path) -> list[TSFSeries]:
    path = Path(path)
    series: list[TSFSeries] = []
    in_data = False
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                in_data = line.lower() == "@data"
                continue
            fields = line.split(":", 5)
            if len(fields) != 6:
                raise ValueError(f"Malformed TSF row {line_number}: expected 6 fields.")
            series_name, city, station, measurement, start_raw, values_raw = fields
            if city not in CITY_KEYS:
                raise ValueError(f"Unsupported city {city!r} on TSF row {line_number}.")
            try:
                pollutant = POLLUTANT_ALIASES[measurement.upper()]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported pollutant {measurement!r} on TSF row {line_number}."
                ) from exc
            tokens = values_raw.split(",")
            mask = np.fromiter((token != "?" for token in tokens), dtype=np.bool_)
            values = np.zeros(len(tokens), dtype=np.float32)
            for index, token in enumerate(tokens):
                if token == "?":
                    continue
                value = float(token)
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite value on TSF row {line_number}, item {index}.")
                values[index] = value
            series.append(
                TSFSeries(
                    series_name=series_name,
                    city=city,
                    station=station,
                    pollutant=pollutant,
                    start_raw=start_raw,
                    start_unix_hour=_parse_clock_hour(start_raw),
                    values=values,
                    mask=mask,
                )
            )
    if not in_data or not series:
        raise ValueError(f"No @data series found in {path}.")
    return series


def _read_coordinates(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"city", "station_id", "longitude", "latitude"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Coordinate CSV must contain {sorted(required)}.")
    return rows


def _split_bounds(total_hours: int) -> dict[str, tuple[int, int]]:
    train_end = total_hours * SPLIT_WEIGHTS[0] // sum(SPLIT_WEIGHTS)
    val_end = total_hours * sum(SPLIT_WEIGHTS[:2]) // sum(SPLIT_WEIGHTS)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total_hours),
    }


def _window_count(start: int, end: int, horizon: int) -> int:
    first_target = max(start, INPUT_WINDOW)
    last_start = end - horizon
    if first_target > last_start:
        return 0
    return (last_start - first_target) // horizon + 1


def _build_graphs(
    city_dir: Path,
    station_ids: list[str],
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> dict[str, object]:
    coords = np.deg2rad(np.column_stack([latitude, longitude]).astype(np.float64))
    lat = coords[:, 0][:, None]
    lon = coords[:, 1][:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    distance = 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    order = np.argsort(distance, axis=1)

    caramoe_k = min(20, len(station_ids) - 1)
    cara_neighbors = order[:, 1 : caramoe_k + 1].astype(np.int64)
    cara_distances = np.take_along_axis(distance, cara_neighbors, axis=1).astype(np.float32)
    caramoe_path = city_dir / f"caramoe_graph_k{caramoe_k}.npz"
    np.savez_compressed(
        caramoe_path,
        neighbor_indices=cara_neighbors,
        distances_km=cara_distances,
        k=np.asarray(caramoe_k, dtype=np.int64),
    )

    aqb_k = min(10, len(station_ids) - 1)
    aqb_neighbors = order[:, 1 : aqb_k + 1]
    query_distances = np.take_along_axis(distance, order[:, : aqb_k + 1], axis=1)
    sigma = float(query_distances.mean())
    if sigma <= 0:
        sigma = 1.0
    adjacency = np.zeros_like(distance, dtype=np.float32)
    for station_index in range(len(station_ids)):
        neighbor_index = aqb_neighbors[station_index]
        neighbor_distance = distance[station_index, neighbor_index]
        adjacency[station_index, neighbor_index] = np.exp(
            -(neighbor_distance**2) / (sigma**2)
        ).astype(np.float32)
    aqb_path = city_dir / f"adj_mx_{aqb_k}.pkl"
    with open(aqb_path, "wb") as handle:
        pickle.dump(
            {
                "sensor_ids": station_ids,
                "id_to_ind": {station: i for i, station in enumerate(station_ids)},
                "adj_mx": adjacency,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return {
        "caramoe_graph": caramoe_path.name,
        "caramoe_k": caramoe_k,
        "aqb_graph": aqb_path.name,
        "aqb_k": aqb_k,
        "aqb_sigma_km": sigma,
    }


def _write_city(
    city_name: str,
    city_series: list[TSFSeries],
    coordinate_rows: list[dict[str, str]],
    output_root: Path,
    source_tsf: Path,
    source_tsf_sha256: str,
) -> dict[str, object]:
    city_key = CITY_KEYS[city_name]
    city_dir = output_root / city_key
    city_dir.mkdir(parents=True, exist_ok=True)

    coordinates = [row for row in coordinate_rows if row["city"].casefold() == city_name.casefold()]
    if "station_index_0based" in coordinates[0]:
        coordinates.sort(key=lambda row: int(row["station_index_0based"]))
    station_ids = [row["station_id"] for row in coordinates]
    station_to_index = {station: index for index, station in enumerate(station_ids)}
    series_stations = {item.station for item in city_series}
    if series_stations != set(station_ids):
        raise ValueError(
            f"{city_name} station mismatch: TSF-only={sorted(series_stations-set(station_ids))}, "
            f"coordinate-only={sorted(set(station_ids)-series_stations)}"
        )

    city_start = min(item.start_unix_hour for item in city_series)
    city_end = max(item.start_unix_hour + len(item.values) for item in city_series)
    total_hours = city_end - city_start
    values = np.zeros((total_hours, len(station_ids), len(POLLUTANTS)), dtype=np.float32)
    masks = np.zeros_like(values, dtype=np.bool_)
    capability = np.zeros((len(station_ids), len(POLLUTANTS)), dtype=np.bool_)
    pollutant_to_index = {name: index for index, name in enumerate(POLLUTANTS)}
    seen: set[tuple[str, str]] = set()
    start_raw_values: list[str] = []
    for item in city_series:
        key = (item.station, item.pollutant)
        if key in seen:
            raise ValueError(f"Duplicate TSF series for {city_name} {key}.")
        seen.add(key)
        station_index = station_to_index[item.station]
        pollutant_index = pollutant_to_index[item.pollutant]
        start = item.start_unix_hour - city_start
        stop = start + len(item.values)
        values[start:stop, station_index, pollutant_index] = item.values
        masks[start:stop, station_index, pollutant_index] = item.mask
        capability[station_index, pollutant_index] = True
        start_raw_values.append(item.start_raw)

    bounds = _split_bounds(total_hours)
    train_start, train_end = bounds["train"]
    train_mask = masks[train_start:train_end]
    train_values = values[train_start:train_end]
    counts = train_mask.sum(axis=(0, 1), dtype=np.int64)
    means = np.zeros(len(POLLUTANTS), dtype=np.float32)
    stds = np.ones(len(POLLUTANTS), dtype=np.float32)
    for pollutant_index in range(len(POLLUTANTS)):
        observed = train_values[..., pollutant_index][train_mask[..., pollutant_index]]
        if observed.size:
            means[pollutant_index] = observed.mean(dtype=np.float64)
            std = observed.std(dtype=np.float64)
            stds[pollutant_index] = float(std) if std > 0 else 1.0

    h5_path = city_dir / f"{city_key}.h5"
    timestamps = city_start + np.arange(total_hours, dtype=np.int64)
    with h5py.File(h5_path, "w") as archive:
        archive.create_dataset("values", data=values, compression="gzip", compression_opts=4)
        archive.create_dataset("masks", data=masks, compression="gzip", compression_opts=4)
        archive.create_dataset("timestamps_unix_hours", data=timestamps)
        archive.create_dataset("structural_capability", data=capability)
        archive.attrs["dataset"] = "KDD Cup 2018 Fresh Air (Zenodo missing version)"
        archive.attrs["city"] = city_name
        archive.attrs["clock_basis"] = "TSF naive local clock encoded as UTC without offset shift"
        archive.attrs["source_tsf"] = str(source_tsf)
        archive.attrs["source_tsf_sha256"] = source_tsf_sha256
        archive.attrs["pollutants_json"] = json.dumps(POLLUTANTS)
        archive.attrs["station_ids_json"] = json.dumps(station_ids)
        archive.attrs["split_ratio"] = "3:1:1"

    metadata_path = city_dir / "metadata.csv"
    with open(metadata_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "station_id",
            "lat",
            "lon",
            "city",
            "site_type",
            "pollutants_in_tsf",
            "coordinate_vintage",
            "coordinate_source_url",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in coordinates:
            writer.writerow(
                {
                    "station_id": row["station_id"],
                    "lat": row["latitude"],
                    "lon": row["longitude"],
                    "city": city_name,
                    "site_type": row.get("site_type", ""),
                    "pollutants_in_tsf": row.get("pollutants_in_tsf", ""),
                    "coordinate_vintage": row.get("coordinate_vintage", ""),
                    "coordinate_source_url": row.get("coordinate_source_url", ""),
                }
            )

    scaler_path = city_dir / "scaler.csv"
    with open(scaler_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pollutant", "mean", "std", "train_observed_count"])
        for pollutant, mean, std, count in zip(POLLUTANTS, means, stds, counts):
            writer.writerow([pollutant, f"{float(mean):.10g}", f"{float(std):.10g}", int(count)])

    train_station_counts = train_mask.sum(axis=0, dtype=np.int64)
    capability_path = city_dir / "capability_train.npz"
    np.savez_compressed(
        capability_path,
        coverage=(train_station_counts / max(train_end - train_start, 1)).astype(np.float32),
        ever_observed=capability.astype(np.float32),
        ever_observed_train=(train_station_counts > 0).astype(np.float32),
        structural_capability=capability.astype(np.float32),
        counts=train_station_counts,
        total_hours=np.asarray(train_end - train_start, dtype=np.int64),
    )

    latitude = np.asarray([float(row["latitude"]) for row in coordinates], dtype=np.float64)
    longitude = np.asarray([float(row["longitude"]) for row in coordinates], dtype=np.float64)
    graph_info = _build_graphs(city_dir, station_ids, latitude, longitude)

    split_manifest: dict[str, object] = {}
    for split_name, (start, end) in bounds.items():
        split_manifest[split_name] = {
            "start_index_inclusive": start,
            "end_index_exclusive": end,
            "hours": end - start,
            "clock_start": _clock_iso(city_start + start),
            "clock_end_exclusive": _clock_iso(city_start + end),
            "fraction": (end - start) / total_hours,
            "window_counts": {
                f"H{horizon}": _window_count(start, end, horizon) for horizon in HORIZONS
            },
        }
    tsf_declared_values = sum(len(item.values) for item in city_series)
    tsf_observed_values = sum(int(item.mask.sum()) for item in city_series)
    city_manifest = {
        "dataset": "KDD Cup 2018 Fresh Air (Zenodo missing version)",
        "city": city_name,
        "city_key": city_key,
        "source_tsf": str(source_tsf),
        "source_tsf_sha256": source_tsf_sha256,
        "h5_file": h5_path.name,
        "h5_sha256": sha256_file(h5_path),
        "station_count": len(station_ids),
        "series_count": len(city_series),
        "total_hours": total_hours,
        "clock_start_normalized": _clock_iso(city_start),
        "clock_end_exclusive": _clock_iso(city_end),
        "source_start_timestamps": sorted(set(start_raw_values)),
        "pollutants": list(POLLUTANTS),
        "structural_capability_count_by_pollutant": {
            pollutant: int(capability[:, index].sum()) for index, pollutant in enumerate(POLLUTANTS)
        },
        "observed_count": int(masks.sum()),
        "possible_structural_count": int(
            sum((city_end - city_start) * capability[:, index].sum() for index in range(6))
        ),
        "tsf_explicit_missing_rate_within_declared_series": float(
            1.0 - tsf_observed_values / max(tsf_declared_values, 1)
        ),
        "aligned_missing_rate_over_declared_capability_cells": float(
            1.0 - masks.sum() / max(total_hours * capability.sum(), 1)
        ),
        "split_ratio": "3:1:1",
        "split_rule": (
            "Chronological target-time split. Forecast targets stay wholly inside one split; "
            "validation/test inputs may use only earlier observed context."
        ),
        "input_window": INPUT_WINDOW,
        "horizons": list(HORIZONS),
        "step_rule": "step equals forecast horizon",
        "splits": split_manifest,
        "scaler_rule": "Per-city, per-pollutant mean/std from observed training cells only",
        "metadata_file": metadata_path.name,
        "scaler_file": scaler_path.name,
        "capability_file": capability_path.name,
        **graph_info,
    }
    write_json(city_dir / "split_manifest.json", city_manifest)
    return city_manifest


def prepare(
    tsf_path: str | Path,
    coordinates_path: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, object]:
    tsf_path = Path(tsf_path).resolve()
    coordinates_path = Path(coordinates_path).resolve()
    output_root = Path(output_root).resolve()
    root_manifest_path = output_root / "protocol_manifest.json"
    if root_manifest_path.exists() and not force:
        raise FileExistsError(
            f"Prepared protocol already exists at {root_manifest_path}. Use --force to rebuild it."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    all_series = read_tsf(tsf_path)
    coordinates = _read_coordinates(coordinates_path)
    source_hash = sha256_file(tsf_path)
    city_manifests = {}
    for city_name in CITY_KEYS:
        selected = [item for item in all_series if item.city == city_name]
        city_manifests[CITY_KEYS[city_name]] = _write_city(
            city_name,
            selected,
            coordinates,
            output_root,
            tsf_path,
            source_hash,
        )
    manifest = {
        "dataset": "KDD Cup 2018 Fresh Air (Zenodo missing version)",
        "source_tsf": str(tsf_path),
        "source_tsf_sha256": source_hash,
        "coordinates": str(coordinates_path),
        "coordinates_sha256": sha256_file(coordinates_path),
        "series_count": len(all_series),
        "cities": city_manifests,
        "split_ratio": "3:1:1",
        "input_window": INPUT_WINDOW,
        "horizons": list(HORIZONS),
        "raw_source_modified": False,
    }
    write_json(root_manifest_path, manifest)
    return manifest


def default_paths() -> tuple[Path, Path, Path]:
    kdd_root = Path(__file__).resolve().parents[2]
    return (
        kdd_root / "raw" / "with_missing_values" / "kdd_cup_2018_dataset_with_missing_values.tsf",
        kdd_root / "metadata" / "stations" / "kdd_cup_2018_station_coordinates.csv",
        kdd_root / "processed" / "3_1_1",
    )


def main(argv: Iterable[str] | None = None) -> None:
    default_tsf, default_coordinates, default_output = default_paths()
    parser = argparse.ArgumentParser(description="Prepare the KDD Fresh Air 3:1:1 protocol.")
    parser.add_argument("--tsf", default=str(default_tsf))
    parser.add_argument("--coordinates", default=str(default_coordinates))
    parser.add_argument("--output-root", default=str(default_output))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = prepare(args.tsf, args.coordinates, args.output_root, force=args.force)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
