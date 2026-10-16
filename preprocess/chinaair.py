from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from preprocess.china_base import (
    POLLUTANTS,
    SPLITS,
    WINDOWS,
    build_adjacency,
    build_metadata,
    build_train_scaler,
    build_year_h5,
    load_and_prepare,
    write_clean_csv,
    write_json,
)
from preprocess.china_zero import (
    apply_per_pollutant_zero_mask,
    build_per_pollutant_zero_audit,
)


DATA_VERSION = "processed_per_pollutant_zero_mask_v4_co_native"
ZERO_MASK_POLICY = "per_pollutant"
CO_UNIT_POLICY = "source_native_no_rescale"
CO_SOURCE_NATIVE_PHYSICAL_LIMIT = 50.0


def restore_source_native_co(selected):
    restored = selected.copy()
    restored["co"] = restored["co"] / 1000.0
    return restored


def mark_h5_protocol(path: Path, zero_cells_masked: int) -> None:
    with h5py.File(path, "r+") as handle:
        handle.attrs["data_version"] = DATA_VERSION
        handle.attrs["zero_mask_policy"] = ZERO_MASK_POLICY
        handle.attrs["all_six_zero_masked"] = True
        handle.attrs["per_pollutant_zero_masked"] = True
        handle.attrs["zero_cells_masked_full_source"] = int(zero_cells_masked)
        handle.attrs["co_unit_conversion"] = "none_source_native"
        handle.attrs["co_value_scale"] = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CHINA3238 with per-pollutant exact-zero masks and CO kept "
            "on its source-native scale (v4)"
        )
    )
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

    selected_legacy_scale, report, all_six_zero_audit = load_and_prepare(source_csv)
    selected_v2_native = restore_source_native_co(selected_legacy_scale)
    per_pollutant_zero_audit, zero_summary = build_per_pollutant_zero_audit(
        selected_v2_native, all_six_zero_audit
    )
    selected = apply_per_pollutant_zero_mask(selected_v2_native)
    if selected[POLLUTANTS].eq(0).to_numpy().any():
        raise AssertionError("Exact-zero pollutant cells remain after v4 masking")

    report["data_version"] = DATA_VERSION
    report["unit_conversion"] = {
        "co": "none; source-native values retained in v4 outputs"
    }
    report["co_unit_policy"] = {
        "name": CO_UNIT_POLICY,
        "source_multiplier": 1.0,
        "legacy_shared_loader_multiplier_undone": 1000.0,
    }
    report["physics_mask"]["limits"]["co"] = CO_SOURCE_NATIVE_PHYSICAL_LIMIT
    report["zero_mask_policy"] = {
        "name": ZERO_MASK_POLICY,
        "masked_cells": int(zero_summary["count_before_masking"]),
        "masked_city_days": int(zero_summary["affected_city_days"]),
        "affected_cities": int(zero_summary["affected_cities"]),
    }
    report["per_pollutant_zero_mask"] = zero_summary
    for pollutant in POLLUTANTS:
        zero_count = int(zero_summary["by_pollutant"][pollutant])
        counts = report["physics_mask"]["invalid_counts"][pollutant]
        counts["exact_zero_before_zero_policy"] = zero_count
        counts["zero_masked_by_policy"] = zero_count
        counts["combined_masked_total_including_zero_policy"] = int(
            counts["masked_total"] + zero_count
        )

    cleaned_csv = output_dir / "CHINA3238_daily_unique.csv"
    write_clean_csv(selected, cleaned_csv)

    all_six_path = output_dir / "all_six_pollutants_zero_audit.csv"
    all_six_output = all_six_zero_audit.rename(columns={"city": "station_id"}).copy()
    all_six_output["date"] = all_six_output["date"].dt.strftime("%Y-%m-%d")
    all_six_output.to_csv(all_six_path, index=False, encoding="utf-8-sig")

    zero_audit_path = output_dir / "per_pollutant_zero_audit.csv"
    zero_output = per_pollutant_zero_audit.rename(
        columns={"city": "station_id"}
    ).copy()
    zero_output["date"] = zero_output["date"].dt.strftime("%Y-%m-%d")
    zero_output.to_csv(zero_audit_path, index=False, encoding="utf-8-sig")

    metadata = build_metadata(selected, data_dir / "selected_nodes_metadata.csv")
    if len(metadata) != 334:
        raise AssertionError(f"Expected 334 retained cities, found {len(metadata)}")
    adjacency_report = build_adjacency(metadata, data_dir / "adj_mx_10.pkl", k=10)
    scaler, scaler_counts = build_train_scaler(selected, data_dir / "scaler.csv")

    h5_reports = []
    for year in [2017, 2018, 2019, 2020, 2021]:
        h5_path = data_dir / f"aq_compact_{year}.h5"
        h5_reports.append(build_year_h5(selected, metadata, year, h5_path))
        mark_h5_protocol(h5_path, zero_summary["count_before_masking"])

    split_config = {
        "dataset": "CHINA3238",
        "data_version": DATA_VERSION,
        "frequency": "daily",
        "parameters": POLLUTANTS,
        "splits": SPLITS,
        "window": WINDOWS,
        "sample_assignment": "assign by forecast target dates; no random time split",
        "normalization": "per-pollutant mean/std from valid training observations only",
        "missing_data": (
            "preserved via boolean masks; every finite exact-zero pollutant cell "
            "is masked independently; no interpolation"
        ),
        "zero_mask_policy": ZERO_MASK_POLICY,
        "co_unit_policy": CO_UNIT_POLICY,
        "metric_protocol": "dual_global_and_legacy_aqb",
    }
    write_json(output_dir / "split_config.json", split_config)

    report["outputs"] = {
        "cleaned_csv": str(cleaned_csv),
        "all_six_pollutants_zero_audit_csv": str(all_six_path),
        "per_pollutant_zero_audit_csv": str(zero_audit_path),
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

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "data_version": DATA_VERSION,
                "cleaned_rows": int(len(selected)),
                "cities": int(len(metadata)),
                "hdf5_years": [2017, 2018, 2019, 2020, 2021],
                "zero_mask_policy": ZERO_MASK_POLICY,
                "zero_cells_masked": int(zero_summary["count_before_masking"]),
                "co_unit_policy": CO_UNIT_POLICY,
                "co_scaler": scaler.loc[
                    scaler["parameter"] == "co", ["mean", "std"]
                ].iloc[0].to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

