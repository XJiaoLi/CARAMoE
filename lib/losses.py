from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
import torch.nn.functional as F


def micro_masked_mae(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.float()
    numerator = ((pred - target).abs() * mask_f).sum()
    denominator = mask_f.sum()
    if denominator <= 0:
        return pred.sum() * 0.0
    return numerator / (denominator + 1e-5)


def balanced_masked_mae(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.float()
    abs_error = (pred - target).abs() * mask_f
    numerator = abs_error.sum(dim=(0, 1, 2))
    denominator = mask_f.sum(dim=(0, 1, 2))
    valid = denominator > 0
    per_pollutant = numerator / denominator.clamp_min(1.0)
    if not valid.any():
        return pred.sum() * 0.0
    return per_pollutant[valid].mean()


def expert_auxiliary_loss(
    expert_predictions: Tensor,
    target: Tensor,
    target_mask: Tensor,
    expert_reliability: Tensor,
) -> Tensor:
    reliability = expert_reliability[:, None].expand_as(expert_predictions)
    mask = target_mask[..., None].float() * reliability
    error = (expert_predictions - target[..., None]).abs() * mask
    num = error.sum(dim=(0, 1, 2, 3))
    den = mask.sum(dim=(0, 1, 2, 3)).clamp_min(1.0)
    return (num / den).mean()


def safe_routing_hinge(
    final_prediction: Tensor,
    self_prediction: Tensor,
    target: Tensor,
    target_mask: Tensor,
    self_reliability: Tensor,
    threshold: float,
    margin: float = 0.0,
) -> Tensor:
    high_support = (self_reliability >= threshold)[:, None]
    valid = target_mask & high_support
    if not valid.any():
        return final_prediction.sum() * 0.0
    final_error = (final_prediction - target).abs()
    self_error = (self_prediction - target).abs()
    hinge = F.relu(final_error - self_error - margin)
    return (hinge * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def route_consistency_loss(
    original_gate: Tensor,
    augmented_gate: Tensor,
    additionally_hidden_channels: Tensor,
) -> Tensor:
    selected = additionally_hidden_channels[:, None].expand_as(original_gate)
    if not selected.any():
        return original_gate.sum() * 0.0
    violation = F.relu(original_gate - augmented_gate)
    return (violation * selected.float()).sum() / selected.float().sum().clamp_min(1.0)


def total_training_loss(
    output: Dict[str, Tensor],
    target: Tensor,
    target_mask: Tensor,
    expert_aux_weight: float,
    safe_weight: float,
    high_reliability_threshold: float,
    safe_margin: float,
    training_variant: str,
) -> tuple[Tensor, Dict[str, Tensor]]:
    variant = training_variant.upper()
    if variant == "A":
        forecast = micro_masked_mae(output["prediction"], target, target_mask)
        zero = forecast.detach().new_zeros(())
        return forecast, {"forecast": forecast.detach(), "expert_aux": zero, "safe": zero}
    if variant != "C":
        raise ValueError(f"Unknown training variant {training_variant!r}; expected A or C.")

    forecast = balanced_masked_mae(output["prediction"], target, target_mask)
    aux = expert_auxiliary_loss(
        output["expert_predictions"], target, target_mask, output["expert_reliability"]
    )
    safe = safe_routing_hinge(
        output["prediction"],
        output["expert_predictions"][..., 0],
        target,
        target_mask,
        output["self_reliability"],
        threshold=high_reliability_threshold,
        margin=safe_margin,
    )
    total = forecast + expert_aux_weight * aux + safe_weight * safe
    return total, {"forecast": forecast.detach(), "expert_aux": aux.detach(), "safe": safe.detach()}
