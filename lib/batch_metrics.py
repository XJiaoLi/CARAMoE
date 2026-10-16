from __future__ import annotations


from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from lib.metrics import POLLUTANTS, inverse_transform


LEGACY_DENOMINATOR_EPSILON = 1e-5
APE_DENOMINATOR_EPSILON = 0.1
METRIC_LEVELS = ("overall", *POLLUTANTS)


def _as_pollutant_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (len(POLLUTANTS),):
        raise ValueError(
            f"{name} must contain {len(POLLUTANTS)} pollutant values, got {result.shape}."
        )
    return result


def _metric_block(
    abs_sum_by_pollutant: Sequence[float] | np.ndarray,
    sq_sum_by_pollutant: Sequence[float] | np.ndarray,
    ape_sum_by_pollutant: Sequence[float] | np.ndarray,
    count_by_pollutant: Sequence[float] | np.ndarray,
    *,
    denominator_epsilon: float = 0.0,
) -> dict[str, dict[str, float]]:

    abs_sum = _as_pollutant_vector(abs_sum_by_pollutant, "abs_sum_by_pollutant")
    sq_sum = _as_pollutant_vector(sq_sum_by_pollutant, "sq_sum_by_pollutant")
    ape_sum = _as_pollutant_vector(ape_sum_by_pollutant, "ape_sum_by_pollutant")
    count = _as_pollutant_vector(count_by_pollutant, "count_by_pollutant")
    if np.any(count < 0):
        raise ValueError("Counts must be non-negative.")

    if denominator_epsilon:
        pollutant_den = count + float(denominator_epsilon)
        overall_den = float(count.sum()) + float(denominator_epsilon)
    else:
        pollutant_den = np.maximum(count, 1.0)
        overall_den = max(float(count.sum()), 1.0)

    pollutant_mse = sq_sum / pollutant_den
    result: dict[str, dict[str, float]] = {
        "overall": {
            "MAE": float(abs_sum.sum() / overall_den),
            "RMSE": math.sqrt(float(sq_sum.sum() / overall_den)),
            "MAPE": float(ape_sum.sum() / overall_den),
            "Count": float(count.sum()),
        }
    }
    for index, pollutant in enumerate(POLLUTANTS):
        result[pollutant] = {
            "MAE": float(abs_sum[index] / pollutant_den[index]),
            "RMSE": math.sqrt(float(pollutant_mse[index])),
            "MAPE": float(ape_sum[index] / pollutant_den[index]),
            "Count": float(count[index]),
        }
    return result


def global_metrics_from_sufficient_statistics(
    abs_sum_by_pollutant: Sequence[float] | np.ndarray,
    sq_sum_by_pollutant: Sequence[float] | np.ndarray,
    ape_sum_by_pollutant: Sequence[float] | np.ndarray,
    count_by_pollutant: Sequence[float] | np.ndarray,
) -> dict[str, dict[str, float]]:

    return _metric_block(
        abs_sum_by_pollutant,
        sq_sum_by_pollutant,
        ape_sum_by_pollutant,
        count_by_pollutant,
        denominator_epsilon=0.0,
    )


@dataclass
class LegacyAQBBatchAccumulator:

    metric_sums: np.ndarray
    batch_count: int

    def __init__(self) -> None:
        self.metric_sums = np.zeros((len(METRIC_LEVELS), 3), dtype=np.float64)
        self.batch_count = 0

    def update_sufficient(
        self,
        abs_sum_by_pollutant: Sequence[float] | np.ndarray,
        sq_sum_by_pollutant: Sequence[float] | np.ndarray,
        ape_sum_by_pollutant: Sequence[float] | np.ndarray,
        count_by_pollutant: Sequence[float] | np.ndarray,
    ) -> None:
        block = _metric_block(
            abs_sum_by_pollutant,
            sq_sum_by_pollutant,
            ape_sum_by_pollutant,
            count_by_pollutant,
            denominator_epsilon=LEGACY_DENOMINATOR_EPSILON,
        )
        for row_index, level in enumerate(METRIC_LEVELS):
            self.metric_sums[row_index] += [
                block[level]["MAE"],
                block[level]["RMSE"],
                block[level]["MAPE"],
            ]
        self.batch_count += 1

    @torch.no_grad()
    def update_tensors(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
        mean: Sequence[float] | np.ndarray,
        std: Sequence[float] | np.ndarray,
    ) -> None:

        prediction_cpu = prediction.detach().cpu().float()
        target_cpu = target.detach().cpu().float()
        mask_cpu = mask.detach().cpu().float()
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32).view(1, 1, 1, -1)
        std_tensor = torch.as_tensor(std, dtype=torch.float32).view(1, 1, 1, -1)
        prediction_real = inverse_transform(prediction_cpu, mean_tensor, std_tensor)
        target_real = inverse_transform(target_cpu, mean_tensor, std_tensor)
        error = prediction_real - target_real
        absolute = error.abs()
        ape = absolute / (target_real.abs() + APE_DENOMINATOR_EPSILON)

        abs_pollutant = (absolute * mask_cpu).sum(dim=(0, 1, 2))
        sq_pollutant = (error.square() * mask_cpu).sum(dim=(0, 1, 2))
        ape_pollutant = (ape * mask_cpu).sum(dim=(0, 1, 2))
        count_pollutant = mask_cpu.sum(dim=(0, 1, 2))

        overall_den = count_pollutant.sum() + LEGACY_DENOMINATOR_EPSILON
        overall_values = torch.stack(
            (
                abs_pollutant.sum() / overall_den,
                torch.sqrt(sq_pollutant.sum() / overall_den),
                ape_pollutant.sum() / overall_den,
            )
        )
        pollutant_den = count_pollutant + LEGACY_DENOMINATOR_EPSILON
        pollutant_values = torch.stack(
            (
                abs_pollutant / pollutant_den,
                torch.sqrt(sq_pollutant / pollutant_den),
                ape_pollutant / pollutant_den,
            ),
            dim=-1,
        )
        values = torch.cat((overall_values[None], pollutant_values), dim=0)
        self.metric_sums += values.double().numpy()
        self.batch_count += 1

    def compute(self) -> dict[str, dict[str, float]]:
        denominator = max(self.batch_count, 1)
        means = self.metric_sums / denominator
        return {
            level: {
                "MAE": float(means[index, 0]),
                "RMSE": float(means[index, 1]),
                "MAPE": float(means[index, 2]),
            }
            for index, level in enumerate(METRIC_LEVELS)
        }


def metrics_from_origin_arrays(
    abs_sum_by_origin_pollutant: np.ndarray,
    sq_sum_by_origin_pollutant: np.ndarray,
    ape_sum_by_origin_pollutant: np.ndarray,
    count_by_origin_pollutant: np.ndarray,
    *,
    batch_size: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], int]:

    arrays = [
        np.asarray(abs_sum_by_origin_pollutant, dtype=np.float64),
        np.asarray(sq_sum_by_origin_pollutant, dtype=np.float64),
        np.asarray(ape_sum_by_origin_pollutant, dtype=np.float64),
        np.asarray(count_by_origin_pollutant, dtype=np.float64),
    ]
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if any(array.ndim != 2 or array.shape[1] != len(POLLUTANTS) for array in arrays):
        raise ValueError("Origin arrays must all have shape [origins, 6].")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("Origin arrays must share the same shape.")

    global_block = global_metrics_from_sufficient_statistics(
        arrays[0].sum(axis=0),
        arrays[1].sum(axis=0),
        arrays[2].sum(axis=0),
        arrays[3].sum(axis=0),
    )
    legacy = LegacyAQBBatchAccumulator()
    for start in range(0, arrays[0].shape[0], batch_size):
        stop = start + batch_size
        legacy.update_sufficient(
            arrays[0][start:stop].sum(axis=0),
            arrays[1][start:stop].sum(axis=0),
            arrays[2][start:stop].sum(axis=0),
            arrays[3][start:stop].sum(axis=0),
        )
    return global_block, legacy.compute(), legacy.batch_count


def origin_records_to_arrays(
    records: Iterable[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    rows = list(records)
    rows.sort(key=lambda record: int(record["sample_index"]))
    sample_index = np.asarray([int(record["sample_index"]) for record in rows], dtype=np.int64)
    if len(np.unique(sample_index)) != len(sample_index):
        raise ValueError("Origin records contain duplicate sample_index values within a group.")
    arrays = []
    for field in ("abs_sum", "sq_sum", "ape_sum", "count"):
        arrays.append(
            np.asarray(
                [_as_pollutant_vector(record[field], field) for record in rows],
                dtype=np.float64,
            )
        )
    return sample_index, arrays[0], arrays[1], arrays[2], arrays[3]
