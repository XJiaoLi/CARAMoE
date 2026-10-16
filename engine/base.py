from __future__ import annotations

import csv
import io
import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from lib.augment import ObservationMaskAugmenter
from lib.config import (
    ExperimentConfig,
    checkpoint_config_fingerprint,
    validate_checkpoint_config,
)
from lib.losses import route_consistency_loss, total_training_loss
from lib.metrics import MetricAccumulator, RegimeMetricAccumulator


CHECKPOINT_VERSION = 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = torch.as_tensor(state["torch"], dtype=torch.uint8).detach().cpu()
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            torch.as_tensor(cuda_state, dtype=torch.uint8).detach().cpu()
            for cuda_state in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def prepare_output_directory(output_dir: str | Path, resume: bool) -> Path | None:
    output_dir = Path(output_dir)
    if resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Cannot resume: output directory does not exist: {output_dir}")
        checkpoint = output_dir / "last.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Cannot resume safely: {checkpoint} is missing. "
                "Legacy best.pt files do not contain complete optimizer/scheduler/RNG state."
            )
        return checkpoint

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to start a new run in non-empty directory: {output_dir}. "
            "Use --resume for a compatible interrupted run or choose a new output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return None


def move_batch(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {
        key: value.to(
            device,
            non_blocking=value.device.type == "cpu" and value.is_pinned(),
        )
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def _checkpoint_snapshot_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _checkpoint_snapshot_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_checkpoint_snapshot_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_snapshot_to_cpu(item) for item in value)
    return value


class CARAMOETrainerBase:
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        resume_checkpoint: str | Path | None = None,
        write_artifacts: bool = True,
    ) -> None:
        config.validate()
        self.config = config
        seed_everything(config.train.seed)
        self.device = resolve_device(config.train.device)
        self.model = model.to(self.device)
        self.mean = mean
        self.std = std
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=config.train.scheduler_patience,
        )
        self.use_amp = bool(config.train.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.augmenter = ObservationMaskAugmenter(
            trailing_block_prob=config.augment.trailing_block_prob,
            trailing_block_min=config.augment.trailing_block_min,
            trailing_block_max=config.augment.trailing_block_max,
            full_channel_drop_prob=config.augment.full_channel_drop_prob,
            station_drop_prob=config.augment.station_drop_prob,
            random_point_drop_prob=config.augment.random_point_drop_prob,
        )

        self.output_dir = Path(config.train.output_dir)
        self.best_val = float("inf")
        self.best_epoch = 0
        self.patience = 0
        self.start_epoch = 1
        self.history: list[Dict[str, float]] = []
        self.best_validation_metrics: Dict[str, float] | None = None

        if write_artifacts:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            config_path = self.output_dir / "config.json"
            if resume_checkpoint is None:
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config.to_dict(), f, indent=2)
            elif not config_path.exists():
                raise FileNotFoundError(f"Resume directory is missing {config_path}.")

        if resume_checkpoint is not None:
            self._resume(resume_checkpoint)

    @property
    def method_label(self) -> str:
        return self.config.method_label

    def _forward(self, batch: Dict[str, Tensor], x: Tensor | None = None, mask: Tensor | None = None):
        return self.model(
            batch["x"] if x is None else x,
            batch["x_mask"] if mask is None else mask,
            batch["x_time"],
            batch["y_time"],
        )

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {
            "loss": 0.0,
            "forecast": 0.0,
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
                loss, components = total_training_loss(
                    output,
                    batch["y"],
                    batch["y_mask"],
                    expert_aux_weight=self.config.train.expert_aux_weight,
                    safe_weight=self.config.train.safe_weight,
                    high_reliability_threshold=self.config.train.high_reliability_threshold,
                    safe_margin=self.config.train.safe_margin_normalized,
                    training_variant=variant,
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
            totals["forecast"] += float(components["forecast"])
            totals["expert_aux"] += float(components["expert_aux"])
            totals["safe"] += float(components["safe"])
            totals["route"] += float(route_loss.detach())
            batches += 1

            if (batch_idx + 1) % self.config.train.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"method={self.method_label} profile={self.config.model.input_profile} "
                    f"epoch={epoch:03d} batch={batch_idx + 1:05d} "
                    f"loss={totals['loss']/batches:.4f} forecast={totals['forecast']/batches:.4f} "
                    f"gate={float(output['external_gate'].mean().detach()):.3f} elapsed={elapsed:.1f}s",
                    flush=True,
                )

        if batches == 0:
            raise RuntimeError("Training loader produced no batches.")
        return {key: value / batches for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, mode: str) -> Dict[str, float]:
        self.model.eval()
        metrics = MetricAccumulator(self.mean, self.std)
        regimes = RegimeMetricAccumulator(self.mean, self.std)
        gate_sum = 0.0
        expert_weight_sum = None
        batches = 0
        max_batches = self.config.train.max_eval_batches

        for batch_idx, raw_batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch(raw_batch, self.device)
            output = self._forward(batch)
            prediction_cpu = output["prediction"].detach().cpu()
            target_cpu = batch["y"].detach().cpu()
            target_mask_cpu = batch["y_mask"].detach().cpu()
            reliability_cpu = output["self_reliability"].detach().cpu()
            metrics.update(prediction_cpu, target_cpu, target_mask_cpu)
            regimes.update(
                prediction_cpu,
                target_cpu,
                target_mask_cpu,
                reliability_cpu,
            )
            gate_sum += float(output["external_gate"].mean())
            weights = output["external_weights"].mean(dim=(0, 1, 2, 3)).detach().cpu()
            expert_weight_sum = weights if expert_weight_sum is None else expert_weight_sum + weights
            batches += 1

        if batches == 0:
            raise RuntimeError(f"{mode} loader produced no batches.")
        result = metrics.compute() | regimes.compute()
        result["MeanExternalGate"] = gate_sum / batches
        if expert_weight_sum is not None:
            names = ("spatial_same", "cross_local", "spatial_cross", "prior")
            for name, value in zip(names, expert_weight_sum / batches):
                result[f"RouteWeight_{name}"] = float(value)
        print(
            f"{mode}: Global_MAE={result['Global_MAE']:.4f}, "
            f"Legacy_AQB_MAE={result['Legacy_AQB_MAE']:.4f}, "
            f"MacroPollutant_MAE={result['MacroPollutant_MAE']:.4f}, "
            f"gate={result['MeanExternalGate']:.3f}",
            flush=True,
        )
        return result

    def _checkpoint_payload(self, epoch: int, val_metrics: Dict[str, float]) -> Dict[str, Any]:
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "amp_scaler": self.scaler.state_dict(),
            "val_metrics": val_metrics,
            "best_val": self.best_val,
            "best_epoch": self.best_epoch,
            "patience": self.patience,
            "history": self.history,
            "rng_state": capture_rng_state(),
            "config": self.config.to_dict(),
            "config_fingerprint": checkpoint_config_fingerprint(self.config),
            "method_label": self.method_label,
            "input_profile": self.config.model.input_profile,
        }

    def _serialize_checkpoint(self, epoch: int, val_metrics: Dict[str, float]) -> bytes:
        payload = _checkpoint_snapshot_to_cpu(self._checkpoint_payload(epoch, val_metrics))
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return buffer.getvalue()

    def _write_serialized_checkpoint(self, filename: str, serialized: bytes) -> Path:
        path = self.output_dir / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "wb") as handle:
            handle.write(serialized)
        temporary.replace(path)
        return path

    def save_checkpoint(self, filename: str, epoch: int, val_metrics: Dict[str, float]) -> Path:
        serialized = self._serialize_checkpoint(epoch, val_metrics)
        return self._write_serialized_checkpoint(filename, serialized)

    def save_checkpoint_copies(
        self,
        filenames: list[str],
        epoch: int,
        val_metrics: Dict[str, float],
    ) -> list[Path]:
        serialized = self._serialize_checkpoint(epoch, val_metrics)
        return [self._write_serialized_checkpoint(filename, serialized) for filename in filenames]

    def _validate_checkpoint_identity(
        self,
        checkpoint: Dict[str, Any],
        checkpoint_path: str | Path,
    ) -> None:
        pass

    def _resume(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint {checkpoint_path} is not a resumable v{CHECKPOINT_VERSION} checkpoint."
            )
        validate_checkpoint_config(
            self.config,
            checkpoint["config"],
            checkpoint.get("config_fingerprint"),
        )
        self._validate_checkpoint_identity(checkpoint, checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint.get("amp_scaler", {}))
        self.best_val = float(checkpoint["best_val"])
        self.best_epoch = int(checkpoint["best_epoch"])
        self.patience = int(checkpoint.get("patience", 0))
        self.history = list(checkpoint.get("history", []))
        self.start_epoch = int(checkpoint["epoch"]) + 1
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        print(
            f"resumed {checkpoint_path} at epoch {checkpoint['epoch']} "
            f"(best epoch={self.best_epoch}, best val={self.best_val:.6f})",
            flush=True,
        )

    def load_model_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "config" not in checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain its experiment config.")
        validate_checkpoint_config(
            self.config,
            checkpoint["config"],
            checkpoint.get("config_fingerprint"),
        )
        self._validate_checkpoint_identity(checkpoint, checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        return checkpoint

    def _write_history(self) -> None:
        if not self.history:
            return
        keys = sorted({key for row in self.history for key in row})
        path = self.output_dir / "history.csv"
        temporary = path.with_suffix(".csv.tmp")
        with open(temporary, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
        temporary.replace(path)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader | None = None):
        last_val_metrics: Dict[str, float] = {}
        for epoch in range(self.start_epoch, self.config.train.epochs + 1):
            train_result = self.train_epoch(train_loader, epoch)
            val_result = self.evaluate(val_loader, "validation")
            last_val_metrics = val_result
            row: Dict[str, float] = {"epoch": float(epoch)}
            row.update({f"train_{key}": value for key, value in train_result.items()})
            row.update({f"val_{key}": value for key, value in val_result.items()})
            self.history.append(row)
            self._write_history()

            val_mae = val_result["Global_MAE"]
            self.scheduler.step(val_mae)
            if val_mae < self.best_val:
                self.best_val = val_mae
                self.best_epoch = epoch
                self.patience = 0
                checkpoint_names = ["best.pt", "last.pt"]
            else:
                self.patience += 1
                checkpoint_names = ["last.pt"]

            checkpoints = self.save_checkpoint_copies(checkpoint_names, epoch, val_result)
            if checkpoint_names[0] == "best.pt":
                print(f"saved {checkpoints[0]} at epoch {epoch}")
            early_patience = self.config.train.early_stopping_patience
            if early_patience is not None and self.patience >= early_patience:
                print(f"early stopping at epoch {epoch}")
                break

        best_path = self.output_dir / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint was not created: {best_path}")
        best_checkpoint = self.load_model_checkpoint(best_path)
        self.best_validation_metrics = dict(best_checkpoint["val_metrics"])
        best_epoch = int(best_checkpoint["epoch"])
        best_val_mae = float(best_checkpoint["val_metrics"]["Global_MAE"])
        del best_checkpoint
        final: Dict[str, Any] = {
            "method_label": self.method_label,
            "training_variant": self.config.train.training_variant,
            "input_profile": self.config.model.input_profile,
            "seed": self.config.train.seed,
            "config_fingerprint": checkpoint_config_fingerprint(self.config),
            "input_window": self.config.data.input_window,
            "output_window": self.config.data.output_window,
            "train_step": self.config.data.train_step,
            "eval_step": self.config.data.eval_step,
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
            "epochs_completed": int(self.history[-1]["epoch"]) if self.history else 0,
        }
        if test_loader is not None:
            test_result = self.evaluate(test_loader, "test")
            final.update({f"test_{key}": value for key, value in test_result.items()})
        path = self.output_dir / "final_metrics.json"
        temporary = path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)
        temporary.replace(path)
        return final
