from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from lib.losses import (
    balanced_masked_mae,
    expert_auxiliary_loss,
    safe_routing_hinge,
    total_training_loss,
)


def aqb_scaled_masked_mae(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    pollutant_std: Tensor,
) -> Tensor:

    std = pollutant_std.to(device=pred.device, dtype=pred.dtype).view(1, 1, 1, -1)
    mask_f = mask.to(dtype=pred.dtype)
    numerator = ((pred - target).abs() * (std + 1e-5) * mask_f).sum()
    denominator = mask_f.sum()
    if denominator <= 0:
        return pred.sum() * 0.0
    return numerator / (denominator + 1e-5)


def total_training_loss_v2(
    output: Dict[str, Tensor],
    target: Tensor,
    target_mask: Tensor,
    expert_aux_weight: float,
    safe_weight: float,
    high_reliability_threshold: float,
    safe_margin: float,
    training_variant: str,
    loss_mode: str,
    pollutant_std: Tensor,
) -> tuple[Tensor, Dict[str, Tensor]]:

    mode = loss_mode.lower()
    if mode == "current":
        return total_training_loss(
            output,
            target,
            target_mask,
            expert_aux_weight=expert_aux_weight,
            safe_weight=safe_weight,
            high_reliability_threshold=high_reliability_threshold,
            safe_margin=safe_margin,
            training_variant=training_variant,
        )
    if mode != "aqb_mix":
        raise ValueError("loss_mode must be 'current' or 'aqb_mix'.")
    if training_variant.upper() != "C":
        raise ValueError("aqb_mix is defined only for CARAMOE training.")

    aqb = aqb_scaled_masked_mae(output["prediction"], target, target_mask, pollutant_std)
    balanced = balanced_masked_mae(output["prediction"], target, target_mask)
    forecast = 0.7 * aqb + 0.3 * balanced
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
    return total, {
        "forecast": forecast.detach(),
        "forecast_aqb": aqb.detach(),
        "forecast_balanced": balanced.detach(),
        "expert_aux": aux.detach(),
        "safe": safe.detach(),
    }
