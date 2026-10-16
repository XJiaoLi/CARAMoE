from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from preprocess.blair import POLLUTANTS, write_json


METRICS = ("MAE", "MSE", "RMSE", "MAPE")


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


class ExactLegacyAccumulator:

    def __init__(self) -> None:
        self.abs_sum = np.zeros(6, dtype=np.float64)
        self.sq_sum = np.zeros(6, dtype=np.float64)
        self.ape_sum = np.zeros(6, dtype=np.float64)
        self.count = np.zeros(6, dtype=np.float64)
        self.legacy_global = {metric: 0.0 for metric in METRICS}
        self.legacy_pollutant = {
            pollutant: {metric: 0.0 for metric in METRICS} for pollutant in POLLUTANTS
        }
        self.legacy_batch_count = 0

    def update(self, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        prediction32 = np.asarray(prediction, dtype=np.float32)
        target32 = np.asarray(target, dtype=np.float32)
        mask_bool = np.asarray(mask, dtype=np.bool_)
        if prediction32.shape != target32.shape or prediction32.shape != mask_bool.shape:
            raise ValueError("Prediction, target, and mask shapes must match.")
        if prediction32.ndim != 4 or prediction32.shape[-1] != 6:
            raise ValueError("Expected [batch, horizon, station, 6] arrays.")

        prediction64 = prediction32.astype(np.float64)
        target64 = target32.astype(np.float64)
        mask64 = mask_bool.astype(np.float64)
        error64 = prediction64 - target64
        axes = (0, 1, 2)
        self.abs_sum += (np.abs(error64) * mask64).sum(axis=axes)
        self.sq_sum += (error64**2 * mask64).sum(axis=axes)
        self.ape_sum += (
            np.abs(error64) / (np.abs(target64) + 0.1) * mask64
        ).sum(axis=axes)
        self.count += mask64.sum(axis=axes)

        error32 = prediction32 - target32
        mask32 = mask_bool.astype(np.float32)
        denominator = np.float32(mask32.sum(dtype=np.float32) + np.float32(1e-5))
        mae = np.float32((np.abs(error32) * mask32).sum(dtype=np.float32) / denominator)
        mse = np.float32(((error32**2) * mask32).sum(dtype=np.float32) / denominator)
        mape = np.float32(
            (np.abs(error32) / (np.abs(target32) + np.float32(0.1)) * mask32).sum(
                dtype=np.float32
            )
            / denominator
        )
        batch_values = {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(np.sqrt(mse, dtype=np.float32)),
            "MAPE": float(mape),
        }
        for metric, value in batch_values.items():
            self.legacy_global[metric] += value

        for pollutant_index, pollutant in enumerate(POLLUTANTS):
            pollutant_error = error32[..., pollutant_index]
            pollutant_target = target32[..., pollutant_index]
            pollutant_mask = mask32[..., pollutant_index]
            pollutant_denominator = np.float32(
                pollutant_mask.sum(dtype=np.float32) + np.float32(1e-5)
            )
            pollutant_mae = np.float32(
                (np.abs(pollutant_error) * pollutant_mask).sum(dtype=np.float32)
                / pollutant_denominator
            )
            pollutant_mse = np.float32(
                ((pollutant_error**2) * pollutant_mask).sum(dtype=np.float32)
                / pollutant_denominator
            )
            pollutant_mape = np.float32(
                (
                    np.abs(pollutant_error)
                    / (np.abs(pollutant_target) + np.float32(0.1))
                    * pollutant_mask
                ).sum(dtype=np.float32)
                / pollutant_denominator
            )
            pollutant_values = {
                "MAE": float(pollutant_mae),
                "MSE": float(pollutant_mse),
                "RMSE": float(np.sqrt(pollutant_mse, dtype=np.float32)),
                "MAPE": float(pollutant_mape),
            }
            for metric, value in pollutant_values.items():
                self.legacy_pollutant[pollutant][metric] += value
        self.legacy_batch_count += 1

    def compute(self) -> dict[str, float | int | None]:
        result: dict[str, float | int | None] = {}
        total_count = float(self.count.sum())
        total_mse = _safe_ratio(float(self.sq_sum.sum()), total_count)
        result["Global_MAE"] = _safe_ratio(float(self.abs_sum.sum()), total_count)
        result["Global_MSE"] = total_mse
        result["Global_RMSE"] = math_sqrt_or_none(total_mse)
        result["Global_MAPE"] = _safe_ratio(float(self.ape_sum.sum()), total_count)
        result["Global_Count"] = int(total_count)
        valid_mae: list[float] = []
        for index, pollutant in enumerate(POLLUTANTS):
            count = float(self.count[index])
            mse = _safe_ratio(float(self.sq_sum[index]), count)
            mae = _safe_ratio(float(self.abs_sum[index]), count)
            result[f"{pollutant}_MAE"] = mae
            result[f"{pollutant}_MSE"] = mse
            result[f"{pollutant}_RMSE"] = math_sqrt_or_none(mse)
            result[f"{pollutant}_MAPE"] = _safe_ratio(float(self.ape_sum[index]), count)
            result[f"{pollutant}_Count"] = int(count)
            if mae is not None:
                valid_mae.append(mae)
        result["MacroPollutant_MAE"] = (
            float(np.mean(valid_mae, dtype=np.float64)) if valid_mae else None
        )

        batch_count = self.legacy_batch_count
        for metric in METRICS:
            result[f"Legacy_AQB_Global_{metric}"] = (
                self.legacy_global[metric] / batch_count if batch_count else None
            )
        result["Legacy_AQB_MAE"] = result["Legacy_AQB_Global_MAE"]
        for pollutant in POLLUTANTS:
            for metric in METRICS:
                result[f"Legacy_AQB_{pollutant}_{metric}"] = (
                    self.legacy_pollutant[pollutant][metric] / batch_count
                    if batch_count
                    else None
                )
        result["Legacy_AQB_Batch_Count"] = batch_count
        return result


def math_sqrt_or_none(value: float | None) -> float | None:
    return float(np.sqrt(value)) if value is not None else None


def read_prediction_artifact(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"].item()))
    required = {"prediction", "target", "mask", "batch_offsets"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"Prediction artifact {path} is missing {sorted(missing)}.")
    return arrays, metadata


def update_from_artifact(
    accumulator: ExactLegacyAccumulator,
    arrays: dict[str, np.ndarray],
) -> None:
    offsets = np.asarray(arrays["batch_offsets"], dtype=np.int64)
    if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(arrays["prediction"]):
        raise ValueError("Invalid batch_offsets in prediction artifact.")
    for start, end in zip(offsets[:-1], offsets[1:]):
        accumulator.update(
            arrays["prediction"][start:end],
            arrays["target"][start:end],
            arrays["mask"][start:end],
        )


def evaluate_artifact(path: str | Path) -> tuple[dict[str, object], dict[str, object]]:
    arrays, metadata = read_prediction_artifact(path)
    accumulator = ExactLegacyAccumulator()
    update_from_artifact(accumulator, arrays)
    return accumulator.compute(), metadata


def aggregate_artifacts(paths: Iterable[str | Path]) -> dict[str, object]:
    records = []
    for path in paths:
        arrays, metadata = read_prediction_artifact(path)
        records.append((Path(path).resolve(), arrays, metadata))
    if len(records) != 2:
        raise ValueError("Joint KDD evaluation requires exactly Beijing and London artifacts.")
    records.sort(key=lambda item: str(item[2].get("city", "")))
    cities = {str(item[2].get("city", "")).casefold() for item in records}
    if cities != {"beijing", "london"}:
        raise ValueError(f"Expected Beijing and London artifacts, got {sorted(cities)}.")
    identity_fields = ("method", "horizon", "seed", "split")
    for field in identity_fields:
        values = {str(item[2].get(field)) for item in records}
        if len(values) != 1:
            raise ValueError(f"Artifact identity mismatch for {field}: {sorted(values)}")

    joint = ExactLegacyAccumulator()
    city_metrics: dict[str, object] = {}
    for path, arrays, metadata in records:
        city_accumulator = ExactLegacyAccumulator()
        update_from_artifact(city_accumulator, arrays)
        update_from_artifact(joint, arrays)
        city_metrics[str(metadata["city"]).casefold()] = {
            "artifact": str(path),
            "metadata": metadata,
            "metrics": city_accumulator.compute(),
        }
    first_metadata = records[0][2]
    return {
        "protocol": {
            "dataset": "KDD Cup 2018 Fresh Air (Zenodo missing version)",
            "city_training": "separate models",
            "joint_exact_aggregation": "pooled observed target cells; never an average of city metrics",
            "joint_legacy_aggregation": (
                "unweighted mean of deterministic city-local evaluation batches, Beijing then London"
            ),
            "legacy_eval_batch_size": first_metadata.get("eval_batch_size"),
            "method": first_metadata.get("method"),
            "horizon": first_metadata.get("horizon"),
            "seed": first_metadata.get("seed"),
            "split": first_metadata.get("split"),
        },
        "joint_metrics": joint.compute(),
        "city_results": city_metrics,
    }


def _write_flat_csv(path: Path, result: dict[str, object]) -> None:
    rows = [{"scope": "joint", **result["joint_metrics"]}]
    for city, city_result in result["city_results"].items():
        rows.append({"scope": city, **city_result["metrics"]})
    keys = ["scope"] + sorted({key for row in rows for key in row if key != "scope"})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pool Beijing and London KDD predictions.")
    parser.add_argument("--beijing", required=True)
    parser.add_argument("--london", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = aggregate_artifacts([args.beijing, args.london])
    output = Path(args.output)
    write_json(output, result)
    _write_flat_csv(output.with_suffix(".csv"), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
