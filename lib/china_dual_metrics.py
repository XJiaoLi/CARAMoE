from __future__ import annotations

import torch

from lib.china_metrics import MaskedMetrics as _BaseMaskedMetrics


PAPER_MAPE_MIN_TARGET = 0.1
LEGACY_MAPE_EPSILON = 0.1
METRIC_NAMES = ("MAE", "MSE", "RMSE", "MAPE")


class DualProtocolMetricAccumulator:

    def __init__(self, metrics):
        self.metrics = metrics
        feature_count = len(metrics.params)

        self.absolute_error_sum = torch.zeros(feature_count, dtype=torch.float64)
        self.squared_error_sum = torch.zeros(feature_count, dtype=torch.float64)
        self.absolute_percentage_error_sum = torch.zeros(
            feature_count, dtype=torch.float64
        )
        self.valid_count = torch.zeros(feature_count, dtype=torch.float64)
        self.mape_valid_count = torch.zeros(feature_count, dtype=torch.float64)

        self.legacy_global_sums = {
            name: torch.zeros((), dtype=torch.float64) for name in METRIC_NAMES
        }
        self.legacy_pollutant_sums = {
            name: torch.zeros(feature_count, dtype=torch.float64)
            for name in METRIC_NAMES
        }
        self.legacy_batch_count = 0

    @torch.no_grad()
    def update(self, preds, labels, mask):
        self.metrics._validate_shapes(preds, labels, mask)
        mask_bool_cpu = mask.detach().to(device="cpu", dtype=torch.bool)

        preds_cpu = preds.detach().to(device="cpu", dtype=torch.float64)
        labels_cpu = labels.detach().to(device="cpu", dtype=torch.float64)
        mask_cpu = mask_bool_cpu.to(dtype=torch.float64)
        preds_real = preds_cpu * (self.metrics.std_cpu + 1e-5) + self.metrics.mean_cpu
        labels_real = labels_cpu * (self.metrics.std_cpu + 1e-5) + self.metrics.mean_cpu
        error = preds_real - labels_real
        reduce_dims = tuple(range(error.ndim - 1))

        self.absolute_error_sum += (error.abs() * mask_cpu).sum(dim=reduce_dims)
        self.squared_error_sum += (error.square() * mask_cpu).sum(dim=reduce_dims)
        paper_mape_mask = mask_bool_cpu & (
            labels_real.abs() >= PAPER_MAPE_MIN_TARGET
        )
        paper_mape_mask_float = paper_mape_mask.to(dtype=torch.float64)
        paper_ape = torch.zeros_like(error)
        paper_ape[paper_mape_mask] = (
            error.abs()[paper_mape_mask] / labels_real.abs()[paper_mape_mask]
        )
        self.absolute_percentage_error_sum += paper_ape.sum(dim=reduce_dims)
        self.valid_count += mask_cpu.sum(dim=reduce_dims)
        self.mape_valid_count += paper_mape_mask_float.sum(dim=reduce_dims)

        legacy_preds = preds.detach().to(device="cpu", dtype=torch.float32)
        legacy_labels = labels.detach().to(device="cpu", dtype=torch.float32)
        legacy_mask = mask_bool_cpu.to(dtype=torch.float32)
        legacy_preds_real = (
            legacy_preds * (self.metrics.legacy_std_cpu + 1e-5)
            + self.metrics.legacy_mean_cpu
        )
        legacy_labels_real = (
            legacy_labels * (self.metrics.legacy_std_cpu + 1e-5)
            + self.metrics.legacy_mean_cpu
        )
        legacy_error = legacy_preds_real - legacy_labels_real

        global_denominator = legacy_mask.sum() + 1e-5
        global_mae = (legacy_error.abs() * legacy_mask).sum() / global_denominator
        global_mse = (legacy_error.square() * legacy_mask).sum() / global_denominator
        global_rmse = torch.sqrt(global_mse)
        global_mape = (
            legacy_error.abs()
            / (legacy_labels_real.abs() + LEGACY_MAPE_EPSILON)
            * legacy_mask
        ).sum() / global_denominator
        for name, value in zip(
            METRIC_NAMES, (global_mae, global_mse, global_rmse, global_mape)
        ):
            self.legacy_global_sums[name] += value.to(dtype=torch.float64)

        for index in range(len(self.metrics.params)):
            pollutant_error = legacy_error[..., index]
            pollutant_labels = legacy_labels_real[..., index]
            pollutant_mask = legacy_mask[..., index]
            pollutant_denominator = pollutant_mask.sum() + 1e-5
            pollutant_mae = (
                pollutant_error.abs() * pollutant_mask
            ).sum() / pollutant_denominator
            pollutant_mse = (
                pollutant_error.square() * pollutant_mask
            ).sum() / pollutant_denominator
            pollutant_rmse = torch.sqrt(pollutant_mse)
            pollutant_mape = (
                pollutant_error.abs()
                / (pollutant_labels.abs() + LEGACY_MAPE_EPSILON)
                * pollutant_mask
            ).sum() / pollutant_denominator
            for name, value in zip(
                METRIC_NAMES,
                (pollutant_mae, pollutant_mse, pollutant_rmse, pollutant_mape),
            ):
                self.legacy_pollutant_sums[name][index] += value.to(
                    dtype=torch.float64
                )
        self.legacy_batch_count += 1

    @staticmethod
    def _safe_ratio(numerator, denominator):
        if float(denominator) <= 0:
            return float("nan")
        return float(numerator / denominator)

    def _legacy_mean(self, value):
        if self.legacy_batch_count <= 0:
            return float("nan")
        return float(value / self.legacy_batch_count)

    def compute(self):
        global_count = self.valid_count.sum()
        global_mape_count = self.mape_valid_count.sum()
        global_mse = self._safe_ratio(self.squared_error_sum.sum(), global_count)
        results = {
            "Global_MAE": self._safe_ratio(
                self.absolute_error_sum.sum(), global_count
            ),
            "Global_MSE": global_mse,
            "Global_RMSE": global_mse ** 0.5,
            "Global_MAPE": self._safe_ratio(
                self.absolute_percentage_error_sum.sum(), global_mape_count
            ),
            "Global_Count": float(global_count),
            "Global_MAPE_Count": float(global_mape_count),
            "Global_MAPE_Valid_Ratio": self._safe_ratio(
                global_mape_count, global_count
            ),
            "Global_MAPE_Excluded_Below_0_1_Count": float(
                global_count - global_mape_count
            ),
            "Legacy_AQB_MAE": self._legacy_mean(
                self.legacy_global_sums["MAE"]
            ),
            "Legacy_AQB_MSE": self._legacy_mean(
                self.legacy_global_sums["MSE"]
            ),
            "Legacy_AQB_RMSE": self._legacy_mean(
                self.legacy_global_sums["RMSE"]
            ),
            "Legacy_AQB_MAPE": self._legacy_mean(
                self.legacy_global_sums["MAPE"]
            ),
            "Legacy_AQB_Batch_Count": float(self.legacy_batch_count),
        }

        for index, pollutant in enumerate(self.metrics.params):
            count = self.valid_count[index]
            mape_count = self.mape_valid_count[index]
            mse = self._safe_ratio(self.squared_error_sum[index], count)
            exact = {
                "MAE": self._safe_ratio(self.absolute_error_sum[index], count),
                "MSE": mse,
                "RMSE": mse ** 0.5,
                "MAPE": self._safe_ratio(
                    self.absolute_percentage_error_sum[index], mape_count
                ),
            }
            for metric_name, value in exact.items():
                results[f"{pollutant}_Global_{metric_name}"] = value
                results[f"{pollutant}_{metric_name}"] = value
            for metric_name in METRIC_NAMES:
                results[f"{pollutant}_Legacy_AQB_{metric_name}"] = (
                    self._legacy_mean(
                        self.legacy_pollutant_sums[metric_name][index]
                    )
                )
            results[f"{pollutant}_Global_Count"] = float(count)
            results[f"{pollutant}_Global_MAPE_Count"] = float(mape_count)
            results[f"{pollutant}_Global_MAPE_Valid_Ratio"] = self._safe_ratio(
                mape_count, count
            )
            results[
                f"{pollutant}_Global_MAPE_Excluded_Below_0_1_Count"
            ] = float(count - mape_count)
            results[f"{pollutant}_Legacy_AQB_Batch_Count"] = float(
                self.legacy_batch_count
            )
            results[f"{pollutant}_Count"] = float(count)
        return results


class DualProtocolMaskedMetrics(_BaseMaskedMetrics):

    def __init__(self, scaler_dict, device, params=None):
        super().__init__(scaler_dict, device, params=params)
        self.legacy_mean_cpu = self.mean.detach().to(
            device="cpu", dtype=torch.float32
        )
        self.legacy_std_cpu = self.std.detach().to(
            device="cpu", dtype=torch.float32
        )

    def accumulator(self):
        return DualProtocolMetricAccumulator(self)


MaskedMetrics = DualProtocolMaskedMetrics



