from __future__ import annotations


import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from lib.metrics import POLLUTANTS, inverse_transform
from engine.base import move_batch


@torch.no_grad()
def batch_origin_statistics(
    prediction: Tensor,
    target: Tensor,
    target_mask: Tensor,
    mean: np.ndarray | Tensor,
    std: np.ndarray | Tensor,
    *,
    self_prediction: Tensor | None = None,
    external_gate: Tensor | None = None,
    external_weights: Tensor | None = None,
) -> dict[str, Tensor]:

    prediction = prediction.detach().double()
    target = target.detach().double()
    mask = target_mask.detach().bool()
    mean_tensor = torch.as_tensor(mean, dtype=torch.float64, device=target.device).reshape(
        1, 1, 1, -1
    )
    std_tensor = torch.as_tensor(std, dtype=torch.float64, device=target.device).reshape(
        1, 1, 1, -1
    )
    prediction_real = inverse_transform(prediction, mean_tensor, std_tensor)
    target_real = inverse_transform(target, mean_tensor, std_tensor)
    mask_f = mask.double()
    error = prediction_real - target_real
    reduce_dims = (1, 2)
    result = {
        "abs_sum_by_pollutant": (error.abs() * mask_f).sum(dim=reduce_dims),
        "sq_sum_by_pollutant": (error.square() * mask_f).sum(dim=reduce_dims),
        "ape_sum_by_pollutant": (
            error.abs() / (target_real.abs() + 0.1) * mask_f
        ).sum(dim=reduce_dims),
        "count_by_pollutant": mask_f.sum(dim=reduce_dims),
    }

    if self_prediction is not None:
        self_real = inverse_transform(
            self_prediction.detach().double(), mean_tensor, std_tensor
        )
        self_error = (self_real - target_real).abs()
        final_error = error.abs()
        result.update(
            {
                "self_abs_sum_by_pollutant": (self_error * mask_f).sum(dim=reduce_dims),
                "final_better_count_by_pollutant": (
                    (final_error < self_error).double() * mask_f
                ).sum(dim=reduce_dims),
                "final_worse_count_by_pollutant": (
                    (final_error > self_error).double() * mask_f
                ).sum(dim=reduce_dims),
                "final_equal_count_by_pollutant": (
                    (final_error == self_error).double() * mask_f
                ).sum(dim=reduce_dims),
            }
        )
    if external_gate is not None:
        result["gate_sum_by_pollutant"] = (
            external_gate.detach().double() * mask_f
        ).sum(dim=reduce_dims)
        result["gate_count_by_pollutant"] = mask_f.sum(dim=reduce_dims)
    if external_weights is not None:
        result["route_weight_sum_by_pollutant_expert"] = (
            external_weights.detach().double() * mask_f[..., None]
        ).sum(dim=reduce_dims)
        result["route_weight_count_by_pollutant"] = mask_f.sum(dim=reduce_dims)
    return {name: value.cpu() for name, value in result.items()}


def _origin_unix_hour(dataset: Any, sample_index: int) -> int:
    year_index, start = dataset.samples[sample_index]
    forecast_start = int(start) + int(dataset.in_len)
    if year_index >= 0:
        info = dataset.years[year_index]
        return int(info.start_unix_hour + forecast_start)
    resolved_year = bisect_right(dataset.year_offsets, forecast_start) - 1
    local_start = forecast_start - dataset.year_offsets[resolved_year]
    return int(dataset.years[resolved_year].start_unix_hour + local_start)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


@torch.no_grad()
def export_origin_sufficient_statistics(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool,
    mean: np.ndarray,
    std: np.ndarray,
    output_path: str | Path,
    split: str,
    years: list[int],
    max_batches: int | None = None,
    test_years_touched: bool = False,
) -> dict[str, Any]:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    rows: dict[str, list[np.ndarray]] = {}
    sample_indices: list[np.ndarray] = []
    batches = 0
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(
                batch["x"], batch["x_mask"], batch["x_time"], batch["y_time"]
            )
        statistics = batch_origin_statistics(
            output["prediction"],
            batch["y"],
            batch["y_mask"],
            mean,
            std,
            self_prediction=output["expert_predictions"][..., 0],
            external_gate=output["external_gate"],
            external_weights=output["external_weights"],
        )
        for name, value in statistics.items():
            rows.setdefault(name, []).append(value.numpy())
        sample_indices.append(batch["sample_index"].detach().cpu().numpy().astype(np.int64))
        batches += 1
    if batches == 0:
        raise RuntimeError(f"{split} loader produced no batches for origin-stat export.")

    arrays = {name: np.concatenate(parts, axis=0) for name, parts in rows.items()}
    arrays["sample_index"] = np.concatenate(sample_indices, axis=0)
    order = np.argsort(arrays["sample_index"], kind="stable")
    arrays = {name: value[order] for name, value in arrays.items()}
    arrays["origin_unix_hour"] = np.asarray(
        [_origin_unix_hour(loader.dataset, int(index)) for index in arrays["sample_index"]],
        dtype=np.int64,
    )
    arrays["abs_sum"] = arrays["abs_sum_by_pollutant"].sum(axis=-1)
    arrays["sq_sum"] = arrays["sq_sum_by_pollutant"].sum(axis=-1)
    arrays["ape_sum"] = arrays["ape_sum_by_pollutant"].sum(axis=-1)
    arrays["count"] = arrays["count_by_pollutant"].sum(axis=-1)

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output_path)

    global_count = float(arrays["count"].sum())
    summary = {
        "schema_version": 1,
        "status": "complete",
        "split": split,
        "years": years,
        "test_years_touched": bool(test_years_touched),
        "rows": int(len(arrays["sample_index"])),
        "batches": batches,
        "pollutants": list(POLLUTANTS),
        "units": "physical concentration units after inverse scaling",
        "bootstrap_unit": "forecast origin; preserve chronological origin_unix_hour ordering",
        "metric_reconstruction": {
            "Global_MAE": "sum(abs_sum) / sum(count)",
            "Global_RMSE": "sqrt(sum(sq_sum) / sum(count))",
            "Global_MAPE": "sum(ape_sum) / sum(count)",
            "pollutant_metrics": "use corresponding *_by_pollutant numerator/count columns",
        },
        "ape_definition": "abs(error) / (abs(target) + 0.1)",
        "npz": output_path.name,
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
        "aggregate_check": {
            "Global_MAE": float(arrays["abs_sum"].sum() / max(global_count, 1.0)),
            "Global_RMSE": float(
                np.sqrt(arrays["sq_sum"].sum() / max(global_count, 1.0))
            ),
            "Global_MAPE": float(arrays["ape_sum"].sum() / max(global_count, 1.0)),
            "count": global_count,
        },
    }
    _write_json_atomic(output_path.with_suffix(".json"), summary)
    return summary

