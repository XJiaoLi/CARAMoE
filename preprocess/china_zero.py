from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

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


DATA_VERSION = "processed_per_pollutant_zero_mask_v3"
ZERO_MASK_POLICY = "per_pollutant"


def exact_zero_cell_mask(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame[POLLUTANTS]
    return values.notna() & values.eq(0)


def apply_per_pollutant_zero_mask(frame: pd.DataFrame) -> pd.DataFrame:
    masked = frame.copy()
    zero_cells = exact_zero_cell_mask(masked)
    masked.loc[:, POLLUTANTS] = masked[POLLUTANTS].mask(zero_cells)
    return masked


def build_per_pollutant_zero_audit(
    selected_after_all_six_mask: pd.DataFrame,
    all_six_zero_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    zero_cells = exact_zero_cell_mask(selected_after_all_six_mask)
    any_remaining_zero = zero_cells.any(axis=1)

    partial = selected_after_all_six_mask.loc[
        any_remaining_zero, ["city", "date", "source", "row_order"]
    ].join(zero_cells.loc[any_remaining_zero])
    partial = partial.melt(
        id_vars=["city", "date", "source", "row_order"],
        value_vars=POLLUTANTS,
        var_name="pollutant",
        value_name="is_exact_zero",
    )
    partial = partial[partial["is_exact_zero"]].drop(columns="is_exact_zero")

    all_six_columns = ["city", "date", "source", "row_order"]
    if all_six_zero_audit.empty:
        all_six = pd.DataFrame(columns=[*all_six_columns, "pollutant"])
    else:
        all_six = all_six_zero_audit.loc[
            all_six_zero_audit.index.repeat(len(POLLUTANTS)), all_six_columns
        ].reset_index(drop=True)
        all_six["pollutant"] = np.tile(POLLUTANTS, len(all_six_zero_audit))

    audit = pd.concat([all_six, partial], ignore_index=True)
    audit["classification"] = "per_pollutant_exact_zero"
    audit["action"] = "mask_only_this_pollutant_false"
    audit = audit.sort_values(
        ["date", "city", "pollutant"], kind="stable"
    ).reset_index(drop=True)

    by_year = audit.assign(year=audit["date"].dt.year).groupby("year").size()
    by_source = audit.groupby("source").size()
    by_pollutant = audit.groupby("pollutant").size().reindex(POLLUTANTS, fill_value=0)
    by_split = {
        name: int(audit["date"].between(config["start"], config["end"]).sum())
        for name, config in SPLITS.items()
    }
    affected_city_days = audit[["city", "date"]].drop_duplicates()
    all_six_cells = len(all_six_zero_audit) * len(POLLUTANTS)
    summary = {
        "rule": "each finite pollutant value exactly equal to zero is invalid independently",
        "classification": "per_pollutant_exact_zero",
        "action": "set only the zero-valued pollutant cell to NaN so its HDF5 mask is false",
        "count_before_masking": int(len(audit)),
        "affected_city_days": int(len(affected_city_days)),
        "affected_cities": int(affected_city_days["city"].nunique()),
        "all_six_zero_city_days": int(len(all_six_zero_audit)),
        "all_six_zero_cells": int(all_six_cells),
        "partial_zero_cells": int(len(audit) - all_six_cells),
        "by_pollutant": {
            str(pollutant): int(count) for pollutant, count in by_pollutant.items()
        },
        "by_year": {str(int(year)): int(count) for year, count in by_year.items()},
        "by_split": by_split,
        "by_source": {str(source): int(count) for source, count in by_source.items()},
    }
    return audit, summary


def mark_h5_zero_policy(path: Path, zero_cells_masked: int) -> None:
    with h5py.File(path, "r+") as handle:
        handle.attrs["data_version"] = DATA_VERSION
        handle.attrs["zero_mask_policy"] = ZERO_MASK_POLICY
        handle.attrs["all_six_zero_masked"] = True
        handle.attrs["per_pollutant_zero_masked"] = True
        handle.attrs["zero_cells_masked_full_source"] = int(zero_cells_masked)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CHINA3238 with every exact-zero pollutant cell masked "
            "independently (v3)"
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

    selected_v2, report, all_six_zero_audit = load_and_prepare(source_csv)
    per_pollutant_zero_audit, per_pollutant_zero_summary = (
        build_per_pollutant_zero_audit(selected_v2, all_six_zero_audit)
    )
    selected = apply_per_pollutant_zero_mask(selected_v2)
    if selected[POLLUTANTS].eq(0).to_numpy().any():
        raise AssertionError("Exact-zero pollutant cells remain after v3 masking")

    report["data_version"] = DATA_VERSION
    report["zero_mask_policy"] = {
        "name": ZERO_MASK_POLICY,
        "masked_cells": int(per_pollutant_zero_summary["count_before_masking"]),
        "masked_city_days": int(per_pollutant_zero_summary["affected_city_days"]),
        "affected_cities": int(per_pollutant_zero_summary["affected_cities"]),
    }
    report["per_pollutant_zero_mask"] = per_pollutant_zero_summary
    for pollutant in POLLUTANTS:
        zero_count = int(per_pollutant_zero_summary["by_pollutant"][pollutant])
        counts = report["physics_mask"]["invalid_counts"][pollutant]
        counts["exact_zero_before_zero_policy"] = zero_count
        counts["zero_masked_by_policy"] = zero_count
        counts["combined_masked_total_including_zero_policy"] = int(
            counts["masked_total"] + zero_count
        )

    cleaned_csv = output_dir / "CHINA3238_daily_unique.csv"
    write_clean_csv(selected, cleaned_csv)

    all_six_zero_audit_path = output_dir / "all_six_pollutants_zero_audit.csv"
    all_six_output = all_six_zero_audit.rename(columns={"city": "station_id"}).copy()
    all_six_output["date"] = all_six_output["date"].dt.strftime("%Y-%m-%d")
    all_six_output.to_csv(all_six_zero_audit_path, index=False, encoding="utf-8-sig")

    per_pollutant_zero_audit_path = output_dir / "per_pollutant_zero_audit.csv"
    per_pollutant_output = per_pollutant_zero_audit.rename(
        columns={"city": "station_id"}
    ).copy()
    per_pollutant_output["date"] = per_pollutant_output["date"].dt.strftime(
        "%Y-%m-%d"
    )
    per_pollutant_output.to_csv(
        per_pollutant_zero_audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = build_metadata(selected, data_dir / "selected_nodes_metadata.csv")
    if len(metadata) != 334:
        raise AssertionError(f"Expected 334 retained cities, found {len(metadata)}")
    adjacency_report = build_adjacency(metadata, data_dir / "adj_mx_10.pkl", k=10)
    scaler, scaler_counts = build_train_scaler(selected, data_dir / "scaler.csv")

    h5_reports = []
    for year in [2017, 2018, 2019, 2020, 2021]:
        h5_path = data_dir / f"aq_compact_{year}.h5"
        h5_reports.append(build_year_h5(selected, metadata, year, h5_path))
        mark_h5_zero_policy(
            h5_path,
            zero_cells_masked=per_pollutant_zero_summary["count_before_masking"],
        )

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
    }
    write_json(output_dir / "split_config.json", split_config)

    report["outputs"] = {
        "cleaned_csv": str(cleaned_csv),
        "all_six_pollutants_zero_audit_csv": str(all_six_zero_audit_path),
        "per_pollutant_zero_audit_csv": str(per_pollutant_zero_audit_path),
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
                "zero_cells_masked": int(
                    per_pollutant_zero_summary["count_before_masking"]
                ),
                "scaler_source": (
                    "2017-01-01 through 2019-12-31, valid observations only"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
