from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


SOURCE_COLUMNS = {
    "city_raw": "城市",
    "date_raw": "日期",
    "city": "标准城市",
    "lon": "经度(WGS84)",
    "lat": "纬度(WGS84)",
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "No2",
    "o3": "O3",
    "so2": "So2",
    "co": "Co",
}

POLLUTANTS = ["pm25", "pm10", "no2", "o3", "so2", "co"]
PHYSICAL_LIMITS = {
    "pm25": 1500.0,
    "pm10": 2000.0,
    "no2": 500.0,
    "o3": 500.0,
    "so2": 500.0,
    "co": 50000.0,
}
EXCLUDED_CITIES = {"伊犁", "大理", "莱芜", "那曲", "韶关"}
SPLITS = {
    "train": {"start": "2017-01-01", "end": "2019-12-31", "years": [2017, 2018, 2019]},
    "validation": {"start": "2020-01-01", "end": "2020-12-31", "years": [2020]},
    "test": {"start": "2021-01-01", "end": "2021-12-31", "years": [2021]},
}
WINDOWS = {"input_days": 30, "forecast_days": [1, 3, 7, 14], "step_days": 1}
ALL_ZERO_RULE = "all six pollutant values are finite and exactly zero"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def all_pollutants_zero_mask(frame: pd.DataFrame) -> pd.Series:
    values = frame[POLLUTANTS]
    return values.notna().all(axis=1) & values.eq(0).all(axis=1)


def _longest_consecutive_runs(audit_rows: pd.DataFrame) -> list[dict]:
    runs = []
    for city, group in audit_rows.sort_values(["city", "date"]).groupby("city"):
        dates = group["date"].sort_values().reset_index(drop=True)
        run_id = dates.diff().dt.days.ne(1).cumsum()
        summary = dates.groupby(run_id).agg(["min", "max", "size"])
        longest = summary.sort_values("size", ascending=False).iloc[0]
        runs.append(
            {
                "city": str(city),
                "max_consecutive_days": int(longest["size"]),
                "run_start": str(longest["min"].date()),
                "run_end": str(longest["max"].date()),
                "all_zero_count": int(len(group)),
            }
        )
    return sorted(runs, key=lambda item: item["max_consecutive_days"], reverse=True)


def audit_all_zero_records(
    selected: pd.DataFrame,
    candidate_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, pd.Series]:
    zero_mask = all_pollutants_zero_mask(selected)
    audit = selected.loc[
        zero_mask, ["city", "date", "source", "row_order"]
    ].copy()
    audit["classification"] = "invalid_zero_sentinel_or_stopped_reporting"
    audit["action"] = "mask_all_six_pollutants_false"
    audit = audit.sort_values(["date", "city"], kind="stable").reset_index(drop=True)

    candidates = candidate_rows.copy()
    candidates["_all_six_zero"] = all_pollutants_zero_mask(candidates)
    linked = candidates.merge(
        audit[["city", "date"]].drop_duplicates(),
        on=["city", "date"],
        how="inner",
    )
    alternatives = (
        linked.groupby(["city", "date"])["_all_six_zero"]
        .agg(["size", "all"])
        .rename(columns={"size": "candidate_rows", "all": "all_candidates_zero"})
    )

    by_year = audit.assign(year=audit["date"].dt.year).groupby("year").size()
    by_source = audit.groupby("source").size()
    by_split = {
        name: int(audit["date"].between(config["start"], config["end"]).sum())
        for name, config in SPLITS.items()
    }
    city_counts = (
        audit.groupby("city")["date"]
        .agg(["size", "min", "max"])
        .sort_values("size", ascending=False)
    )
    city_summary = [
        {
            "city": str(city),
            "count": int(row["size"]),
            "first_date": str(row["min"].date()),
            "last_date": str(row["max"].date()),
        }
        for city, row in city_counts.iterrows()
    ]
    summary = {
        "rule": ALL_ZERO_RULE,
        "classification": "invalid_zero_sentinel_or_stopped_reporting",
        "decision_basis": (
            "Simultaneous exact zeros across all six pollutants include long "
            "consecutive city-level runs (up to hundreds of days), which is "
            "incompatible with physically plausible daily concentrations."
        ),
        "action": "set all six pollutant values to NaN so HDF5 masks are false",
        "count_before_masking": int(zero_mask.sum()),
        "affected_cities": int(audit["city"].nunique()),
        "by_year": {str(int(year)): int(count) for year, count in by_year.items()},
        "by_split": by_split,
        "by_source": {str(source): int(count) for source, count in by_source.items()},
        "duplicate_keys": int((alternatives["candidate_rows"] > 1).sum()),
        "keys_with_nonzero_alternative": int(
            (~alternatives["all_candidates_zero"]).sum()
        ),
        "city_summary": city_summary,
        "longest_consecutive_runs": _longest_consecutive_runs(audit),
    }
    return audit, summary, zero_mask


def load_and_prepare(source_csv: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    required = list(SOURCE_COLUMNS.values())
    df = pd.read_csv(source_csv, usecols=required, low_memory=False)
    df = df.rename(columns={value: key for key, value in SOURCE_COLUMNS.items()})

    raw_rows = len(df)
    raw_cities = int(df["city"].nunique())
    raw_date = df["date_raw"].astype(str).str.strip()
    df["date"] = pd.to_datetime(
        raw_date.str.replace("/", "-", regex=False),
        format="%Y-%m-%d",
        errors="coerce",
    )
    invalid_dates = int(df["date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Found {invalid_dates} unparseable dates")

    df["source"] = np.where(
        raw_date.str.contains("/", regex=False), "nationwide", "city_file"
    )
    df["source_priority"] = np.where(df["source"].eq("city_file"), 1, 0)
    df["row_order"] = np.arange(len(df), dtype=np.int64)

    for pollutant in POLLUTANTS:
        df[pollutant] = pd.to_numeric(df[pollutant], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    keys = ["city", "date"]
    key_sizes = df.groupby(keys, sort=False).size()
    duplicate_sizes = key_sizes[key_sizes > 1]
    duplicate_rows = df[df.duplicated(keys, keep=False)]
    conflict_nunique = duplicate_rows.groupby(keys, sort=False)[POLLUTANTS].nunique(
        dropna=True
    )
    conflict_groups = int((conflict_nunique > 1).any(axis=1).sum())

    df = df[~df["city"].isin(EXCLUDED_CITIES)].copy()
    excluded_rows = raw_rows - len(df)

    df = df.sort_values(keys + ["source_priority", "row_order"], kind="stable")
    selected = df.drop_duplicates(keys, keep="last").copy()

    selected["co"] = selected["co"] * 1000.0
    all_zero_audit, all_zero_summary, all_zero_mask = audit_all_zero_records(
        selected, df
    )

    invalid_by_pollutant: dict[str, dict[str, int]] = {}
    for pollutant in POLLUTANTS:
        values = selected[pollutant]
        missing = values.isna()
        below_zero = values < 0
        above_limit = values > PHYSICAL_LIMITS[pollutant]
        invalid = missing | below_zero | above_limit
        invalid_by_pollutant[pollutant] = {
            "missing_before_physics_filter": int(missing.sum()),
            "negative": int(below_zero.sum()),
            "above_physical_limit": int(above_limit.sum()),
            "masked_total": int(invalid.sum()),
            "combined_masked_total_including_all_six_zero": int(
                invalid.sum() + all_zero_mask.sum()
            ),
        }
        selected.loc[invalid, pollutant] = np.nan
    selected.loc[all_zero_mask, POLLUTANTS] = np.nan

    coord_variants = selected.groupby("city")[["lat", "lon"]].nunique(dropna=False)
    inconsistent_coords = coord_variants[(coord_variants > 1).any(axis=1)]
    if not inconsistent_coords.empty:
        raise ValueError(
            "Cities with inconsistent coordinates: "
            + ", ".join(inconsistent_coords.index.astype(str).tolist())
        )
    if selected[["city", "lat", "lon"]].drop_duplicates()[["lat", "lon"]].isna().any().any():
        raise ValueError("Missing city coordinates after deduplication")

    selected = selected.sort_values(["date", "city"], kind="stable").reset_index(drop=True)
    if selected.duplicated(["city", "date"]).any():
        raise AssertionError("Duplicate city-date keys remain after cleaning")

    report = {
        "source": {
            "path": str(source_csv.resolve()),
            "sha256": sha256_file(source_csv),
            "raw_rows": raw_rows,
            "raw_cities": raw_cities,
            "date_min": str(selected["date"].min().date()),
            "date_max": str(selected["date"].max().date()),
            "slash_date_rows": int(raw_date.str.contains("/", regex=False).sum()),
            "dash_date_rows": int(raw_date.str.contains("-", regex=False).sum()),
        },
        "deduplication": {
            "duplicate_city_date_groups_before_exclusion": int(len(duplicate_sizes)),
            "duplicate_extra_rows_before_exclusion": int((duplicate_sizes - 1).sum()),
            "conflicting_duplicate_groups_before_exclusion": conflict_groups,
            "selection_rule": "prefer city_file; otherwise keep last nationwide row in source order",
        },
        "city_filter": {
            "excluded_cities": sorted(EXCLUDED_CITIES),
            "excluded_rows": int(excluded_rows),
            "retained_cities": int(selected["city"].nunique()),
        },
        "unit_conversion": {"co": "source value multiplied by 1000"},
        "all_six_zero_mask": all_zero_summary,
        "physics_mask": {
            "limits": PHYSICAL_LIMITS,
            "invalid_counts": invalid_by_pollutant,
        },
        "cleaned": {
            "unique_city_date_rows": int(len(selected)),
            "unique_cities": int(selected["city"].nunique()),
        },
    }
    return selected, report, all_zero_audit


def write_clean_csv(selected: pd.DataFrame, output_path: Path) -> None:
    clean = selected[["city", "date", *POLLUTANTS]].copy()
    clean = clean.rename(columns={"city": "station_id"})
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    clean.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6g",
    )


def build_metadata(selected: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    metadata = (
        selected[["city", "lat", "lon"]]
        .drop_duplicates("city")
        .sort_values("city", kind="stable")
        .rename(columns={"city": "station_id"})
        .reset_index(drop=True)
    )
    metadata.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    return metadata


def build_adjacency(metadata: pd.DataFrame, output_path: Path, k: int = 10) -> dict:
    station_ids = metadata["station_id"].astype(str).tolist()
    coords_rad = np.radians(metadata[["lat", "lon"]].to_numpy(dtype=np.float64))
    tree = BallTree(coords_rad, metric="haversine")
    dist, ind = tree.query(coords_rad, k=k + 1)
    dist_km = dist * 6371.0
    sigma = float(np.mean(dist_km))

    adj_mx = np.zeros((len(metadata), len(metadata)), dtype=np.float32)
    for node_idx in range(len(metadata)):
        for neighbor_pos in range(1, k + 1):
            neighbor_idx = int(ind[node_idx, neighbor_pos])
            distance = float(dist_km[node_idx, neighbor_pos])
            adj_mx[node_idx, neighbor_idx] = np.exp(-(distance**2) / (sigma**2))

    payload = {
        "sensor_ids": station_ids,
        "id_to_ind": {station_id: idx for idx, station_id in enumerate(station_ids)},
        "adj_mx": adj_mx,
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "k": k,
        "sigma_km": sigma,
        "shape": list(adj_mx.shape),
        "nonzero_edges": int(np.count_nonzero(adj_mx)),
        "directed": True,
    }


def build_year_h5(
    selected: pd.DataFrame,
    metadata: pd.DataFrame,
    year: int,
    output_path: Path,
) -> dict:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    station_ids = metadata["station_id"].astype(str).tolist()
    station_to_idx = {station_id: idx for idx, station_id in enumerate(station_ids)}

    values = np.full((len(dates), len(station_ids), len(POLLUTANTS)), np.nan, dtype=np.float32)
    masks = np.zeros_like(values, dtype=np.bool_)

    year_df = selected[selected["date"].dt.year.eq(year)]
    if year_df.duplicated(["city", "date"]).any():
        raise AssertionError(f"Duplicate city-date keys found for {year}")

    time_idx = (year_df["date"] - dates[0]).dt.days.to_numpy(dtype=np.int64)
    node_idx = year_df["city"].map(station_to_idx).to_numpy(dtype=np.int64)
    year_values = year_df[POLLUTANTS].to_numpy(dtype=np.float32)
    year_masks = np.isfinite(year_values)
    values[time_idx, node_idx, :] = year_values
    masks[time_idx, node_idx, :] = year_masks

    chunk_days = min(31, len(dates))
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(
            "values",
            data=values,
            dtype="f4",
            compression="lzf",
            chunks=(chunk_days, 1, len(POLLUTANTS)),
        )
        handle.create_dataset(
            "masks",
            data=masks,
            dtype="b1",
            compression="lzf",
            chunks=(chunk_days, 1, len(POLLUTANTS)),
        )
        utf8_string = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            "station_ids",
            data=np.asarray(station_ids, dtype=object),
            dtype=utf8_string,
        )
        handle.create_dataset("params", data=np.asarray(POLLUTANTS, dtype="S8"))
        handle.create_dataset("dates", data=np.asarray(dates.strftime("%Y-%m-%d"), dtype="S10"))
        handle.attrs["frequency"] = "daily"
        handle.attrs["year"] = year
        handle.attrs["co_unit_conversion"] = "source_x1000"
        handle.attrs["all_six_zero_masked"] = True

    valid_counts = masks.sum(axis=(0, 1))
    total_cells = masks.shape[0] * masks.shape[1]
    return {
        "year": year,
        "shape": list(values.shape),
        "valid_counts": {p: int(valid_counts[i]) for i, p in enumerate(POLLUTANTS)},
        "coverage_ratio": {
            p: float(valid_counts[i] / total_cells) for i, p in enumerate(POLLUTANTS)
        },
    }


def build_train_scaler(selected: pd.DataFrame, output_path: Path) -> tuple[pd.DataFrame, dict]:
    train = selected[selected["date"].between(SPLITS["train"]["start"], SPLITS["train"]["end"])]
    rows = []
    counts = {}
    for pollutant in POLLUTANTS:
        valid = train[pollutant].dropna().to_numpy(dtype=np.float64)
        if valid.size == 0:
            raise ValueError(f"No valid training observations for {pollutant}")
        mean = float(valid.mean())
        std = float(valid.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid training standard deviation for {pollutant}: {std}")
        rows.append({"parameter": pollutant, "mean": mean, "std": std})
        counts[pollutant] = int(valid.size)
    scaler = pd.DataFrame(rows)
    scaler.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    return scaler, counts


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CHINA3238 for daily AirQualityBench experiments")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_csv = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source_csv.exists():
        raise FileNotFoundError(source_csv)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    selected, report, all_zero_audit = load_and_prepare(source_csv)
    cleaned_csv = output_dir / "CHINA3238_daily_unique.csv"
    write_clean_csv(selected, cleaned_csv)
    all_zero_audit_path = output_dir / "all_six_pollutants_zero_audit.csv"
    audit_output = all_zero_audit.rename(columns={"city": "station_id"}).copy()
    audit_output["date"] = audit_output["date"].dt.strftime("%Y-%m-%d")
    audit_output.to_csv(all_zero_audit_path, index=False, encoding="utf-8-sig")

    metadata = build_metadata(selected, data_dir / "selected_nodes_metadata.csv")
    if len(metadata) != 334:
        raise AssertionError(f"Expected 334 retained cities, found {len(metadata)}")

    adjacency_report = build_adjacency(metadata, data_dir / "adj_mx_10.pkl", k=10)
    scaler, scaler_counts = build_train_scaler(selected, data_dir / "scaler.csv")

    h5_reports = []
    for year in [2017, 2018, 2019, 2020, 2021]:
        h5_reports.append(
            build_year_h5(
                selected,
                metadata,
                year,
                data_dir / f"aq_compact_{year}.h5",
            )
        )

    split_config = {
        "dataset": "CHINA3238",
        "frequency": "daily",
        "parameters": POLLUTANTS,
        "splits": SPLITS,
        "window": WINDOWS,
        "sample_assignment": "assign by forecast target dates; no random time split",
        "normalization": "per-pollutant mean/std from valid training observations only",
        "missing_data": (
            "preserved via boolean masks; all-six-pollutant exact-zero city-days "
            "are masked; no interpolation"
        ),
    }
    write_json(output_dir / "split_config.json", split_config)

    report["outputs"] = {
        "cleaned_csv": str(cleaned_csv),
        "all_six_pollutants_zero_audit_csv": str(all_zero_audit_path),
        "hdf5_years": [2017, 2018, 2019, 2020, 2021],
        "station_order": metadata["station_id"].astype(str).tolist(),
        "parameter_order": POLLUTANTS,
        "scaler": scaler.to_dict(orient="records"),
        "scaler_valid_counts": scaler_counts,
        "adjacency": adjacency_report,
        "hdf5": h5_reports,
        "split_config": split_config,
    }
    write_json(output_dir / "quality_report.json", report)

    print(json.dumps({
        "output_dir": str(output_dir),
        "cleaned_rows": int(len(selected)),
        "cities": int(len(metadata)),
        "hdf5_years": [2017, 2018, 2019, 2020, 2021],
        "all_six_zero_records_masked": int(len(all_zero_audit)),
        "scaler_source": "2017-01-01 through 2019-12-31, valid observations only",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


