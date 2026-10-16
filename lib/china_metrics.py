import torch


class MaskedMetricAccumulator:

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

    @torch.no_grad()
    def update(self, preds, labels, mask):
        self.metrics._validate_shapes(preds, labels, mask)
        preds_cpu = preds.detach().to(device="cpu", dtype=torch.float64)
        labels_cpu = labels.detach().to(device="cpu", dtype=torch.float64)
        mask_bool_cpu = mask.detach().to(device="cpu", dtype=torch.bool)
        mask_cpu = mask_bool_cpu.to(dtype=torch.float64)

        preds_real = preds_cpu * (self.metrics.std_cpu + 1e-5) + self.metrics.mean_cpu
        labels_real = labels_cpu * (self.metrics.std_cpu + 1e-5) + self.metrics.mean_cpu
        error = preds_real - labels_real
        reduce_dims = tuple(range(error.ndim - 1))

        self.absolute_error_sum += (error.abs() * mask_cpu).sum(dim=reduce_dims)
        self.squared_error_sum += (error.square() * mask_cpu).sum(dim=reduce_dims)
        mape_mask = mask_bool_cpu & labels_real.ne(0)
        mape_mask_float = mape_mask.to(dtype=torch.float64)
        absolute_percentage_error = torch.zeros_like(error)
        absolute_percentage_error[mape_mask] = (
            error.abs()[mape_mask] / labels_real.abs()[mape_mask]
        )
        self.absolute_percentage_error_sum += absolute_percentage_error.sum(
            dim=reduce_dims
        )
        self.valid_count += mask_cpu.sum(dim=reduce_dims)
        self.mape_valid_count += mape_mask_float.sum(dim=reduce_dims)

    @staticmethod
    def _safe_ratio(numerator, denominator):
        if float(denominator) <= 0:
            return float("nan")
        return float(numerator / denominator)

    def compute(self):
        global_count = self.valid_count.sum()
        global_mape_count = self.mape_valid_count.sum()
        global_mse = self._safe_ratio(self.squared_error_sum.sum(), global_count)
        results = {
            "Global_MAE": self._safe_ratio(self.absolute_error_sum.sum(), global_count),
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
            "Global_MAPE_Excluded_Zero_Count": float(
                global_count - global_mape_count
            ),
        }
        for index, name in enumerate(self.metrics.params):
            count = self.valid_count[index]
            mape_count = self.mape_valid_count[index]
            mse = self._safe_ratio(self.squared_error_sum[index], count)
            results[f"{name}_MAE"] = self._safe_ratio(
                self.absolute_error_sum[index], count
            )
            results[f"{name}_MSE"] = mse
            results[f"{name}_RMSE"] = mse ** 0.5
            results[f"{name}_MAPE"] = self._safe_ratio(
                self.absolute_percentage_error_sum[index], mape_count
            )
            results[f"{name}_Count"] = float(count)
            results[f"{name}_MAPE_Count"] = float(mape_count)
            results[f"{name}_MAPE_Valid_Ratio"] = self._safe_ratio(
                mape_count, count
            )
            results[f"{name}_MAPE_Excluded_Zero_Count"] = float(
                count - mape_count
            )
        return results


class MaskedMetrics:
    def __init__(self, scaler_dict, device, params=None):
        means = torch.tensor(scaler_dict["mean"], dtype=torch.float32)
        stds = torch.tensor(scaler_dict["std"], dtype=torch.float32)
        if means.ndim != 1 or stds.ndim != 1 or means.shape != stds.shape:
            raise ValueError("Scaler mean/std must be one-dimensional and have equal length")
        if not torch.isfinite(means).all() or not torch.isfinite(stds).all() or torch.any(stds <= 0):
            raise ValueError("Scaler values must be finite and every std must be positive")
        self.mean = means.view(1, 1, 1, -1).to(device)
        self.std = stds.view(1, 1, 1, -1).to(device)
        self.mean_cpu = means.to(dtype=torch.float64).view(1, 1, 1, -1)
        self.std_cpu = stds.to(dtype=torch.float64).view(1, 1, 1, -1)
        self.params = list(params or ["pm25", "pm10", "no2", "o3", "so2", "co"])
        if len(self.params) != means.numel():
            raise ValueError("Parameter names must match scaler feature count")

    def inverse_transform(self, values):
        return values * (self.std + 1e-5) + self.mean

    @staticmethod
    def _validate_shapes(preds, labels, mask):
        if preds.shape != labels.shape or labels.shape != mask.shape:
            raise ValueError(
                f"Prediction, label, and mask shapes must match; got "
                f"{preds.shape}, {labels.shape}, {mask.shape}"
            )

    def accumulator(self):
        return MaskedMetricAccumulator(self)

    def compute_all(self, preds, labels, mask):
        accumulator = self.accumulator()
        accumulator.update(preds, labels, mask)
        return accumulator.compute()

    @staticmethod
    def _mask_as_float(mask, reference):
        return mask.to(device=reference.device, dtype=reference.dtype)

    def _masked_mae(self, preds, labels, mask):
        mask_float = self._mask_as_float(mask, preds)
        loss = torch.abs(preds - labels) * mask_float
        return torch.sum(loss) / (torch.sum(mask_float) + 1e-5)

    def _masked_mse(self, preds, labels, mask):
        mask_float = self._mask_as_float(mask, preds)
        loss = ((preds - labels) ** 2) * mask_float
        return torch.sum(loss) / (torch.sum(mask_float) + 1e-5)

    def _masked_rmse(self, preds, labels, mask):
        return torch.sqrt(self._masked_mse(preds, labels, mask))

    def _masked_mape(self, preds, labels, mask, eps=0.1):
        mask_float = self._mask_as_float(mask, preds)
        loss = (torch.abs(preds - labels) / (torch.abs(labels) + eps)) * mask_float
        return torch.sum(loss) / (torch.sum(mask_float) + 1e-5)



