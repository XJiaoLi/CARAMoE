from __future__ import annotations


import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model.caramoe import CARAMOE


EXTERNAL_EXPERT_NAMES: tuple[str, ...] = (
    "spatial_same",
    "cross_local",
    "spatial_cross",
    "prior",
)


@dataclass(frozen=True)
class F1AblationSpec:
    name: str
    router_mode: str = "adaptive_factorized"
    use_availability_context: bool = True
    use_capability_context: bool = True
    force_no_trust_cap: bool = False
    disable_safe_loss: bool = False
    disable_route_consistency: bool = False
    description: str = ""


F1_ABLATION_SPECS: Mapping[str, F1AblationSpec] = {
    "full": F1AblationSpec(
        "full",
        description="Frozen F1-full reference model.",
    ),
    "c1_self_only": F1AblationSpec(
        "c1_self_only",
        router_mode="self_only",
        description="Use only the residual self forecast; external intervention is zero.",
    ),
    "c2_uniform_external": F1AblationSpec(
        "c2_uniform_external",
        router_mode="uniform_external",
        description="Uniformly allocate active external experts while retaining the learned gate.",
    ),
    "c3_static_routing": F1AblationSpec(
        "c3_static_routing",
        router_mode="static_external",
        force_no_trust_cap=True,
        description=(
            "Use learned horizon-specific global external weights and a global static gate; "
            "continuous sample A/C/R/content is ignored, binary expert validity is retained, "
            "and no dynamic trust cap is applied."
        ),
    ),
    "c4_direct_moe": F1AblationSpec(
        "c4_direct_moe",
        router_mode="direct_moe",
        force_no_trust_cap=True,
        description="Replace factorized residual routing with one five-expert softmax.",
    ),
    "c5_availability_only": F1AblationSpec(
        "c5_availability_only",
        use_capability_context=False,
        description="Retain current availability but remove long-term capability conditioning.",
    ),
    "c6_capability_only": F1AblationSpec(
        "c6_capability_only",
        use_availability_context=False,
        description="Retain long-term capability but remove explicit availability features.",
    ),
    "c7_neither_availability_nor_capability": F1AblationSpec(
        "c7_neither_availability_nor_capability",
        use_availability_context=False,
        use_capability_context=False,
        description="Remove both explicit availability and capability conditioning.",
    ),
    "c8_no_trust_cap": F1AblationSpec(
        "c8_no_trust_cap",
        force_no_trust_cap=True,
        description="Keep factorized routing but remove the structural trust cap.",
    ),
    "c9_no_safe_loss": F1AblationSpec(
        "c9_no_safe_loss",
        disable_safe_loss=True,
        description="Set the high-support safe-routing loss weight to zero.",
    ),
    "c10_no_route_consistency": F1AblationSpec(
        "c10_no_route_consistency",
        disable_route_consistency=True,
        description="Set the missingness route-consistency loss weight to zero.",
    ),
}

_VARIANT_ALIASES = {
    "c1": "c1_self_only",
    "c2": "c2_uniform_external",
    "c3": "c3_static_routing",
    "c4": "c4_direct_moe",
    "c5": "c5_availability_only",
    "c6": "c6_capability_only",
    "c7": "c7_neither_availability_nor_capability",
    "c8": "c8_no_trust_cap",
    "c9": "c9_no_safe_loss",
    "c10": "c10_no_route_consistency",
}

LEAVE_ONE_OUT_GROUPS: Mapping[str, tuple[str, ...]] = {
    "none": (),
    "spatial_same": ("spatial_same",),
    "cross_local": ("cross_local",),
    "spatial_cross": ("spatial_cross",),
    "prior": ("prior",),
    "both_cross": ("cross_local", "spatial_cross"),
    "all_external": EXTERNAL_EXPERT_NAMES,
}


def canonical_f1_variant(name: str) -> str:
    canonical = name.strip().lower().replace("-", "_")
    canonical = _VARIANT_ALIASES.get(canonical, canonical)
    if canonical not in F1_ABLATION_SPECS:
        choices = ", ".join(F1_ABLATION_SPECS)
        raise ValueError(f"Unknown F1 ablation {name!r}; choose one of: {choices}.")
    return canonical


def resolve_excluded_experts(
    leave_one_out: str = "none",
    extra_excluded: Iterable[str] = (),
) -> tuple[str, ...]:
    group = leave_one_out.strip().lower().replace("-", "_")
    if group not in LEAVE_ONE_OUT_GROUPS:
        choices = ", ".join(LEAVE_ONE_OUT_GROUPS)
        raise ValueError(f"Unknown leave-one-out group {leave_one_out!r}; choose: {choices}.")
    requested = set(LEAVE_ONE_OUT_GROUPS[group]) | {
        item.strip().lower() for item in extra_excluded if item.strip()
    }
    unknown = requested.difference(EXTERNAL_EXPERT_NAMES)
    if unknown:
        raise ValueError(f"Unknown external experts: {sorted(unknown)}.")
    return tuple(name for name in EXTERNAL_EXPERT_NAMES if name in requested)


def _masked_normalize(weights: Tensor, valid: Tensor) -> Tensor:
    weights = weights * valid.to(dtype=weights.dtype)
    denominator = weights.sum(dim=-1, keepdim=True)
    return torch.where(
        denominator > 0,
        weights / denominator.clamp_min(1e-8),
        torch.zeros_like(weights),
    )


def _masked_softmax(logits: Tensor, valid: Tensor) -> Tensor:
    masked = logits.masked_fill(~valid, -1e4)
    return _masked_normalize(torch.softmax(masked, dim=-1), valid)


class _QueryContextMask(nn.Module):

    def __init__(self, base: nn.Module, use_availability: bool, use_capability: bool) -> None:
        super().__init__()
        self.base = base
        self.use_availability = bool(use_availability)
        self.use_capability = bool(use_capability)

    def forward(self, context: Tensor) -> Tensor:
        masked = context.clone()
        if not self.use_availability:
            masked[..., -7:-2] = 0
        if not self.use_capability:
            masked[..., -2:] = 0
        return self.base(masked)


class _CapabilityContextMask(nn.Module):

    def __init__(self, base: nn.Module, trailing_capability_dim: int = 2) -> None:
        super().__init__()
        self.base = base
        self.trailing_capability_dim = int(trailing_capability_dim)

    def forward(self, context: Tensor, *args, **kwargs):
        masked = context.clone()
        masked[..., -self.trailing_capability_dim :] = 0
        return self.base(masked, *args, **kwargs)


class _SpatialAvailabilityMask(nn.Module):

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(
        self,
        query: Tensor,
        states: Tensor,
        source_reliability: Tensor,
        availability: Tensor,
    ):
        return self.base(
            query,
            states,
            source_reliability,
            torch.zeros_like(availability),
        )


class _LagFreshnessTransform(nn.Module):

    def __init__(self, base: nn.Module, mode: str) -> None:
        super().__init__()
        if mode not in {"decay", "binary"}:
            raise ValueError("freshness_mode must be 'decay' or 'binary'.")
        self.base = base
        self.mode = mode

    def forward(
        self, query: Tensor, snapshots: Tensor, snapshot_freshness: Tensor
    ):
        freshness = (
            snapshot_freshness
            if self.mode == "decay"
            else (snapshot_freshness > 1e-6).to(dtype=snapshot_freshness.dtype)
        )
        return self.base(query, snapshots, freshness)


class _F1AblationRouter(nn.Module):

    def __init__(
        self,
        base: nn.Module,
        mode: str,
        output_horizon: int,
        use_availability: bool,
        use_capability: bool,
        excluded_experts: Sequence[str],
    ) -> None:
        super().__init__()
        if mode not in {
            "adaptive_factorized",
            "self_only",
            "uniform_external",
            "static_external",
            "direct_moe",
        }:
            raise ValueError(f"Unsupported router ablation mode {mode!r}.")
        self.base = base
        self.mode = mode
        self.use_availability = bool(use_availability)
        self.use_capability = bool(use_capability)
        excluded = torch.tensor(
            [name in excluded_experts for name in EXTERNAL_EXPERT_NAMES], dtype=torch.bool
        )
        self.register_buffer("excluded_external_mask", excluded, persistent=True)
        self.static_external_logits = (
            nn.Parameter(torch.zeros(output_horizon, len(EXTERNAL_EXPERT_NAMES)))
            if mode == "static_external"
            else None
        )
        self.static_gate_logits = (
            nn.Parameter(torch.full((output_horizon,), -1.0))
            if mode == "static_external"
            else None
        )

    def _masked_context(self, availability: Tensor, capability: Tensor) -> tuple[Tensor, Tensor]:
        availability_context = availability if self.use_availability else torch.zeros_like(availability)
        capability_context = capability if self.use_capability else torch.zeros_like(capability)
        return availability_context, capability_context

    def _external_valid(self, expert_reliability: Tensor, horizon: int) -> Tensor:
        valid = expert_reliability[:, None, :, :, 1:] > 1e-6
        active = ~self.excluded_external_mask
        return valid.expand(-1, horizon, -1, -1, -1) & active.view(1, 1, 1, 1, -1)

    def _direct_moe(
        self,
        query: Tensor,
        expert_features: Tensor,
        expert_reliability: Tensor,
        availability: Tensor,
        capability: Tensor,
    ) -> tuple[Tensor, Tensor]:
        q = self.base.q_proj(query)
        e = self.base.e_proj(expert_features)
        base_scores = torch.einsum("bnpr,bnper->bnpe", q, e) / math.sqrt(q.shape[-1])
        base_scores = base_scores + torch.log(expert_reliability.clamp_min(1e-6))
        context_input = torch.cat(
            [query, availability, capability, expert_reliability], dim=-1
        )
        route_context = self.base.route_context(context_input)
        horizon_logits = torch.einsum(
            "bnpr,hre->bhnpe", route_context, self.base.route_horizon
        )
        logits = (
            base_scores[:, None]
            + horizon_logits
            + self.base.horizon_bias[None, :, None, None, :]
        )
        valid = expert_reliability[:, None] > 1e-6
        expert_active = torch.cat(
            [
                torch.ones(1, dtype=torch.bool, device=logits.device),
                ~self.excluded_external_mask,
            ]
        )
        valid = valid & expert_active.view(1, 1, 1, 1, -1)
        probabilities = _masked_softmax(logits, valid)
        gate = 1.0 - probabilities[..., 0]
        external_weights = torch.where(
            gate[..., None] > 1e-8,
            probabilities[..., 1:] / gate[..., None].clamp_min(1e-8),
            torch.zeros_like(probabilities[..., 1:]),
        )
        return external_weights, gate

    def forward(
        self,
        query: Tensor,
        expert_features: Tensor,
        expert_reliability: Tensor,
        availability: Tensor,
        capability: Tensor,
    ) -> tuple[Tensor, Tensor]:
        availability_context, capability_context = self._masked_context(
            availability, capability
        )
        horizon = self.base.output_horizon
        routing_reliability = expert_reliability
        if bool(self.excluded_external_mask.any()):
            routing_reliability = expert_reliability.clone()
            routing_reliability[..., 1:] = routing_reliability[..., 1:].masked_fill(
                self.excluded_external_mask.view(1, 1, 1, -1), 0.0
            )
        external_valid = self._external_valid(routing_reliability, horizon)

        if self.mode == "direct_moe":
            return self._direct_moe(
                query,
                expert_features,
                routing_reliability,
                availability_context,
                capability_context,
            )

        if self.mode == "self_only" or bool(self.excluded_external_mask.all()):
            weights = torch.zeros_like(external_valid, dtype=query.dtype)
            gate = torch.zeros(external_valid.shape[:-1], dtype=query.dtype, device=query.device)
            return weights, gate

        if self.mode == "static_external":
            assert self.static_external_logits is not None
            assert self.static_gate_logits is not None
            logits = self.static_external_logits.view(1, horizon, 1, 1, -1)
            weights = _masked_softmax(logits.expand(external_valid.shape), external_valid)
            active_any = external_valid.any(dim=-1)
            gate = torch.sigmoid(self.static_gate_logits).view(1, horizon, 1, 1)
            gate = gate.expand(active_any.shape)
            return weights, gate * active_any.to(dtype=gate.dtype)

        learned_weights, learned_gate = self.base(
            query,
            expert_features,
            routing_reliability,
            availability_context,
            capability_context,
        )
        if self.mode == "uniform_external":
            weights = _masked_normalize(
                torch.ones_like(learned_weights), external_valid
            )
        else:
            weights = _masked_normalize(learned_weights, external_valid)
        active_any = external_valid.any(dim=-1)
        return weights, learned_gate * active_any.to(dtype=learned_gate.dtype)


class CARAMOEAblation(CARAMOE):

    VARIANT_NAME = "CARAMOE_ablation"

    def __init__(
        self,
        coords_deg: np.ndarray,
        neighbor_indices: np.ndarray,
        distances_km: np.ndarray,
        capability_coverage: np.ndarray,
        capability_ever: np.ndarray,
        output_horizon: int,
        *args,
        ablation: str = "full",
        leave_one_out: str = "none",
        excluded_experts: Sequence[str] = (),
        trust_floor: float = 0.1,
        freshness_mode: str = "decay",
        **kwargs,
    ) -> None:
        self.ablation_name = canonical_f1_variant(ablation)
        self.ablation_spec = F1_ABLATION_SPECS[self.ablation_name]
        self.leave_one_out = leave_one_out.strip().lower().replace("-", "_")
        self.excluded_experts = resolve_excluded_experts(
            self.leave_one_out, excluded_experts
        )
        self.requested_trust_floor = float(trust_floor)
        self.freshness_mode = freshness_mode.strip().lower()
        if self.freshness_mode not in {"decay", "binary"}:
            raise ValueError("freshness_mode must be 'decay' or 'binary'.")
        effective_trust_floor = (
            1.0 if self.ablation_spec.force_no_trust_cap else float(trust_floor)
        )
        super().__init__(
            coords_deg,
            neighbor_indices,
            distances_km,
            capability_coverage,
            capability_ever,
            output_horizon,
            *args,
            trust_floor=effective_trust_floor,
            **kwargs,
        )

        context_changed = not (
            self.ablation_spec.use_availability_context
            and self.ablation_spec.use_capability_context
        )
        router_changed = (
            self.ablation_spec.router_mode != "adaptive_factorized"
            or bool(self.excluded_experts)
            or context_changed
        )
        if context_changed:
            self.query_encoder = _QueryContextMask(
                self.query_encoder,
                self.ablation_spec.use_availability_context,
                self.ablation_spec.use_capability_context,
            )
            if not self.ablation_spec.use_capability_context:
                self.prior_encoder = _CapabilityContextMask(self.prior_encoder)
                if self.baseline_router is not None:
                    self.baseline_router = _CapabilityContextMask(self.baseline_router)
            if (
                not self.ablation_spec.use_availability_context
                and self.use_multiscale_spatial
            ):
                self.spatial_aggregator = _SpatialAvailabilityMask(
                    self.spatial_aggregator
                )
        if self.cross_mode == "lagged" and self.freshness_mode != "decay":
            self.cross_aggregator = _LagFreshnessTransform(
                self.cross_aggregator, self.freshness_mode
            )
        if router_changed:
            self.router = _F1AblationRouter(
                self.router,
                mode=self.ablation_spec.router_mode,
                output_horizon=output_horizon,
                use_availability=self.ablation_spec.use_availability_context,
                use_capability=self.ablation_spec.use_capability_context,
                excluded_experts=self.excluded_experts,
            )

    def experiment_metadata(self) -> dict[str, object]:
        return super().experiment_metadata() | {
            "experiment_family": "CARAMOE_ablation",
            "ablation": self.ablation_name,
            "ablation_description": self.ablation_spec.description,
            "router_mode": self.ablation_spec.router_mode,
            "use_availability_context": self.ablation_spec.use_availability_context,
            "use_capability_context": self.ablation_spec.use_capability_context,
            "requested_trust_floor": self.requested_trust_floor,
            "effective_trust_floor": self.trust_floor,
            "freshness_mode": self.freshness_mode,
            "leave_one_out": self.leave_one_out,
            "excluded_experts": list(self.excluded_experts),
            "expert_names": list(self.EXPERT_NAMES),
            "availability_ablation_boundary": (
                "Explicit 5-D target availability is masked from query/router and the "
                "multiscale selector; native masks and expert source reliability remain "
                "for mask-safe evidence construction. Except for C3/C4/C8, the structural "
                "self-reliability trust cap is retained."
            ),
            "capability_ablation_boundary": (
                "The 2-D long-term capability is masked from query/router and static prior."
            ),
            "leave_one_out_boundary": (
                "The removed expert is masked before router logits, renormalized out of the "
                "external simplex, and excluded from auxiliary supervision. Expert tensors "
                "are still computed to preserve the shared architecture. Removing cross_local "
                "alone leaves its upstream use inside spatial_cross; both_cross removes both "
                "cross-derived routes."
            ),
            "frozen_parent_model": "model/caramoe.py",
        }

    def forward(
        self, x: Tensor, x_mask: Tensor, x_time: Tensor, y_time: Tensor
    ) -> Dict[str, Tensor]:
        output = super().forward(x, x_mask, x_time, y_time)
        auxiliary_excluded = (
            EXTERNAL_EXPERT_NAMES
            if self.ablation_spec.router_mode == "self_only"
            else self.excluded_experts
        )
        if auxiliary_excluded:
            reliability = output["expert_reliability"].clone()
            indices = [self.EXPERT_NAMES.index(name) for name in auxiliary_excluded]
            reliability[..., indices] = 0
            output["expert_reliability"] = reliability

        gate = output["external_gate"]
        output["fusion_weights"] = torch.cat(
            [(1.0 - gate)[..., None], gate[..., None] * output["external_weights"]],
            dim=-1,
        )
        active = torch.tensor(
            [name not in auxiliary_excluded for name in EXTERNAL_EXPERT_NAMES],
            dtype=output["prediction"].dtype,
            device=output["prediction"].device,
        )
        output["active_external_expert_mask"] = active
        return output
