from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from torch import Tensor

POLLUTANTS = ("pm25", "pm10", "no2", "o3", "so2", "co")


def inverse_transform(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return x * (std + 1e-5) + mean


class MetricAccumulator:

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.as_tensor(mean, dtype=torch.float64).view(1, 1, 1, 6)
        self.std = torch.as_tensor(std, dtype=torch.float64).view(1, 1, 1, 6)
        self.legacy_mean = torch.as_tensor(mean, dtype=torch.float32).view(1, 1, 1, 6)
        self.legacy_std = torch.as_tensor(std, dtype=torch.float32).view(1, 1, 1, 6)
        self.abs_sum = torch.zeros(6, dtype=torch.float64)
        self.sq_sum = torch.zeros(6, dtype=torch.float64)
        self.ape_sum = torch.zeros(6, dtype=torch.float64)
        self.count = torch.zeros(6, dtype=torch.float64)
        self.legacy_global_mae_sum = 0.0
        self.legacy_batch_count = 0

    @torch.no_grad()
    def update(self, pred: Tensor, target: Tensor, mask: Tensor) -> None:
        pred_cpu = pred.detach().cpu()
        target_cpu = target.detach().cpu()
        mask_cpu = mask.detach().cpu()

        legacy_pred_real = inverse_transform(pred_cpu.float(), self.legacy_mean, self.legacy_std)
        legacy_target_real = inverse_transform(target_cpu.float(), self.legacy_mean, self.legacy_std)
        legacy_mask = mask_cpu.float()
        legacy_batch_mae = (
            (legacy_pred_real - legacy_target_real).abs() * legacy_mask
        ).sum() / (legacy_mask.sum() + 1e-5)
        self.legacy_global_mae_sum += float(legacy_batch_mae)
        self.legacy_batch_count += 1

        pred = pred_cpu.double()
        target = target_cpu.double()
        mask_f = mask_cpu.double()
        pred_real = inverse_transform(pred, self.mean, self.std)
        target_real = inverse_transform(target, self.mean, self.std)
        error = pred_real - target_real
        absolute_error = error.abs() * mask_f
        self.abs_sum += absolute_error.sum(dim=(0, 1, 2))
        self.sq_sum += (error.square() * mask_f).sum(dim=(0, 1, 2))
        self.ape_sum += (error.abs() / (target_real.abs() + 0.1) * mask_f).sum(dim=(0, 1, 2))
        self.count += mask_f.sum(dim=(0, 1, 2))

    def compute(self) -> Dict[str, float]:
        den = self.count.clamp_min(1.0)
        mae = self.abs_sum / den
        mse = self.sq_sum / den
        mape = self.ape_sum / den
        result: Dict[str, float] = {}
        valid = self.count > 0
        global_count = self.count.sum().clamp_min(1.0)
        global_mse = self.sq_sum.sum() / global_count
        result["Global_MAE"] = float(self.abs_sum.sum() / global_count)
        result["Global_MSE"] = float(global_mse)
        result["Global_RMSE"] = float(torch.sqrt(global_mse))
        result["Global_MAPE"] = float(self.ape_sum.sum() / self.count.sum().clamp_min(1.0))
        result["Legacy_AQB_MAE"] = self.legacy_global_mae_sum / max(self.legacy_batch_count, 1)
        result["MacroPollutant_MAE"] = float(mae[valid].mean()) if valid.any() else float("nan")
        for i, name in enumerate(POLLUTANTS):
            result[f"{name}_MAE"] = float(mae[i])
            result[f"{name}_MSE"] = float(mse[i])
            result[f"{name}_RMSE"] = float(torch.sqrt(mse[i]))
            result[f"{name}_MAPE"] = float(mape[i])
            result[f"{name}_Count"] = float(self.count[i])
        return result


class RegimeMetricAccumulator:

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.as_tensor(mean, dtype=torch.float64).view(1, 1, 1, 6)
        self.std = torch.as_tensor(std, dtype=torch.float64).view(1, 1, 1, 6)
        self.sums = defaultdict(float)
        self.counts = defaultdict(float)

    @torch.no_grad()
    def update(self, pred: Tensor, target: Tensor, target_mask: Tensor, self_reliability: Tensor) -> None:
        pred = pred.detach().cpu().double()
        target = target.detach().cpu().double()
        target_mask = target_mask.detach().cpu().bool()
        reliability = self_reliability.detach().cpu().double()[:, None]
        pred_real = inverse_transform(pred, self.mean, self.std)
        target_real = inverse_transform(target, self.mean, self.std)
        error = (pred_real - target_real).abs()
        bins = {
            "no_support": reliability <= 1e-8,
            "low_support": (reliability > 1e-8) & (reliability < 0.35),
            "medium_support": (reliability >= 0.35) & (reliability < 0.75),
            "high_support": reliability >= 0.75,
        }
        for name, condition in bins.items():
            valid = target_mask & condition
            self.sums[name] += float((error * valid.float()).sum())
            self.counts[name] += float(valid.float().sum())

    def compute(self) -> Dict[str, float]:
        return {
            f"Regime_{name}_MAE": self.sums[name] / max(self.counts[name], 1.0)
            for name in ("no_support", "low_support", "medium_support", "high_support")
        } | {
            f"Regime_{name}_Count": self.counts[name]
            for name in ("no_support", "low_support", "medium_support", "high_support")
        }
