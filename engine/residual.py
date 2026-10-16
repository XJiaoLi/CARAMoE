from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.losses import route_consistency_loss
from lib.losses_aligned import total_training_loss_v2
from engine.base import CARAMOETrainerBase, CHECKPOINT_VERSION, move_batch


class CARAMOETrainer(CARAMOETrainerBase):

    def __init__(
        self,
        model,
        config,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        loss_mode: str,
        v2_metadata: dict[str, object],
        resume_checkpoint: str | Path | None = None,
        write_artifacts: bool = True,
    ) -> None:
        self.loss_mode = loss_mode.lower()
        if self.loss_mode not in {"current", "aqb_mix"}:
            raise ValueError("loss_mode must be 'current' or 'aqb_mix'.")
        self.v2_metadata = dict(v2_metadata) | {"loss_mode": self.loss_mode}
        self.pollutant_std = torch.as_tensor(std, dtype=torch.float32)
        super().__init__(
            model,
            config,
            mean,
            std,
            resume_checkpoint=resume_checkpoint,
            write_artifacts=write_artifacts,
        )

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {
            "loss": 0.0,
            "forecast": 0.0,
            "forecast_aqb": 0.0,
            "forecast_balanced": 0.0,
            "expert_aux": 0.0,
            "safe": 0.0,
            "route": 0.0,
        }
        batches = 0
        start_time = time.time()
        max_batches = self.config.train.max_train_batches
        variant = self.config.train.training_variant.upper()

        for batch_idx, raw_batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch(raw_batch, self.device)
            x, x_mask = batch["x"], batch["x_mask"]
            if self.config.augment.enabled:
                augmented = self.augmenter(x, x_mask)
                model_x, model_mask = augmented.values, augmented.mask
            else:
                augmented = None
                model_x, model_mask = x, x_mask

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self._forward(batch, model_x, model_mask)
                loss, components = total_training_loss_v2(
                    output,
                    batch["y"],
                    batch["y_mask"],
                    expert_aux_weight=self.config.train.expert_aux_weight,
                    safe_weight=self.config.train.safe_weight,
                    high_reliability_threshold=self.config.train.high_reliability_threshold,
                    safe_margin=self.config.train.safe_margin_normalized,
                    training_variant=variant,
                    loss_mode=self.loss_mode,
                    pollutant_std=self.pollutant_std,
                )
                route_loss = loss.new_zeros(())
                if (
                    variant == "C"
                    and augmented is not None
                    and self.config.train.route_consistency_weight > 0
                    and batch_idx % max(1, self.config.train.run_consistency_every) == 0
                ):
                    with torch.no_grad():
                        original_output = self._forward(batch)
                    route_loss = route_consistency_loss(
                        original_output["external_gate"].detach(),
                        output["external_gate"],
                        augmented.additionally_hidden_channels,
                    )
                    loss = loss + self.config.train.route_consistency_weight * route_loss

            self.scaler.scale(loss).backward()
            if self.config.train.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            totals["loss"] += float(loss.detach())
            for name, value in components.items():
                totals[name] += float(value)
            totals["route"] += float(route_loss.detach())
            batches += 1

            if (batch_idx + 1) % self.config.train.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"method={self.method_label} loss={self.loss_mode} "
                    f"profile={self.config.model.input_profile} epoch={epoch:03d} "
                    f"batch={batch_idx + 1:05d} loss={totals['loss']/batches:.4f} "
                    f"forecast={totals['forecast']/batches:.4f} "
                    f"gate={float(output['external_gate'].mean().detach()):.3f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        if batches == 0:
            raise RuntimeError("Training loader produced no batches.")
        return {key: value / batches for key, value in totals.items()}

    def _checkpoint_payload(self, epoch: int, val_metrics: Dict[str, float]) -> Dict[str, Any]:
        payload = super()._checkpoint_payload(epoch, val_metrics)
        payload["v2_metadata"] = self.v2_metadata
        return payload

    def _validate_v2_checkpoint(self, checkpoint: Dict[str, Any], checkpoint_path: str | Path) -> None:
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(f"Checkpoint {checkpoint_path} has an incompatible version.")
        saved = checkpoint.get("v2_metadata")
        if saved != self.v2_metadata:
            raise ValueError(
                "V2 checkpoint identity mismatch: "
                f"saved={saved!r}, requested={self.v2_metadata!r}."
            )

    def _validate_checkpoint_identity(
        self,
        checkpoint: Dict[str, Any],
        checkpoint_path: str | Path,
    ) -> None:
        self._validate_v2_checkpoint(checkpoint, checkpoint_path)
