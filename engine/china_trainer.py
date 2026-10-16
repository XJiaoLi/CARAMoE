import glob
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from lib.china_dual_metrics import DualProtocolMaskedMetrics as MaskedMetrics
from lib.augment import ObservationMaskAugmenter
from lib.losses import route_consistency_loss
from lib.losses_aligned import total_training_loss_v2


DEFAULT_F1_TRAINING_CONFIG = {
    "loss_mode": "aqb_mix",
    "training_variant": "C",
    "expert_aux_weight": 0.15,
    "safe_weight": 0.1,
    "route_consistency_weight": 0.05,
    "high_reliability_threshold": 0.75,
    "safe_margin_normalized": 0.0,
    "run_consistency_every": 4,
    "grad_clip": 5.0,
    "amp": True,
    "augment": {
        "enabled": True,
        "trailing_block_prob": 0.25,
        "trailing_block_min": 3,
        "trailing_block_max": 24,
        "full_channel_drop_prob": 0.05,
        "station_drop_prob": 0.02,
        "random_point_drop_prob": 0.03,
    },
}


class Trainer:
    def __init__(self, model, scaler_dict, args, device, params=None):
        self.model = model.to(device)
        self.device = device
        self.args = args
        self.optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=5, factor=0.5
        )
        self.metrics_tool = MaskedMetrics(scaler_dict, device, params=params)

        self.use_full_f1_objective = bool(
            getattr(self.model, "supports_full_f1_objective", False)
        )
        self.f1_training_config = self._resolve_f1_training_config()
        self.pollutant_std = torch.as_tensor(
            scaler_dict["std"], dtype=torch.float32, device=device
        )
        self.use_amp = bool(
            self.use_full_f1_objective
            and self.f1_training_config["amp"]
            and device.type == "cuda"
        )
        try:
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except AttributeError:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        augment = self.f1_training_config["augment"]
        self.f1_augmenter = ObservationMaskAugmenter(
            trailing_block_prob=augment["trailing_block_prob"],
            trailing_block_min=augment["trailing_block_min"],
            trailing_block_max=augment["trailing_block_max"],
            full_channel_drop_prob=augment["full_channel_drop_prob"],
            station_drop_prob=augment["station_drop_prob"],
            random_point_drop_prob=augment["random_point_drop_prob"],
        )

        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if getattr(args, "resume_dir", ""):
            self.exp_dir = Path(args.resume_dir)
        else:
            folder_name = (
                f"{args.dataset}_{args.model}_daily_"
                f"in{args.w_in}_out{args.w_out}_step{args.step}_{self.run_timestamp}"
            )
            self.exp_dir = Path(args.output_dir) / folder_name

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.exp_dir / "results.csv"
        self.final_test_path = self.exp_dir / "final_test_results.csv"
        self.summary_path = self.exp_dir / "experiment_summary.json"
        self.last_checkpoint_path = self.exp_dir / "last_checkpoint.pth"
        self.last_checkpoint_meta_path = self.exp_dir / "last_checkpoint.json"
        self.epoch_logs = []
        self.best_model_path = None
        self.best_val_mae = float("inf")
        self.best_epoch = None
        self._save_effective_config()
        self._save_model_metadata()

    def _resolve_f1_training_config(self):
        config = dict(DEFAULT_F1_TRAINING_CONFIG)
        config["augment"] = dict(DEFAULT_F1_TRAINING_CONFIG["augment"])
        supplied = dict(getattr(self.args, "caramoe_training_config", {}) or {})
        supplied_augment = dict(supplied.pop("augment", {}) or {})
        config.update(supplied)
        config["augment"].update(supplied_augment)
        if config["loss_mode"] not in {"current", "aqb_mix"}:
            raise ValueError("caramoe training loss_mode must be current or aqb_mix")
        if str(config["training_variant"]).upper() != "C":
            raise ValueError("CHINA3238 CARAMOE full training requires training_variant C")
        for name in (
            "expert_aux_weight", "safe_weight", "route_consistency_weight", "grad_clip"
        ):
            if float(config[name]) < 0:
                raise ValueError(f"caramoe training {name} cannot be negative")
        if int(config["run_consistency_every"]) <= 0:
            raise ValueError("run_consistency_every must be positive")
        return config

    def _save_effective_config(self):
        config = {}
        for key, value in vars(self.args).items():
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            config[key] = value
        config["resolved_caramoe_training_config"] = self.f1_training_config
        config["uses_full_f1_training_objective"] = self.use_full_f1_objective

        config_path = self.exp_dir / "effective_config.json"
        if getattr(self.args, "resume_dir", "") and config_path.exists():
            config_path = self.exp_dir / f"resume_config_{self.run_timestamp}.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

    def _save_model_metadata(self):
        if not hasattr(self.model, "experiment_metadata"):
            return
        metadata = dict(self.model.experiment_metadata())
        if self.use_full_f1_objective:
            metadata["resolved_training_config"] = self.f1_training_config
            metadata["metric_aggregation"] = (
                "global error numerator / global valid-mask denominator"
            )
        metadata_path = self.exp_dir / "model_metadata.json"
        if getattr(self.args, "resume_dir", "") and metadata_path.exists():
            metadata_path = self.exp_dir / f"resume_model_metadata_{self.run_timestamp}.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    def _load_torch_file(self, path):
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def _restore_history_from_csv(self, up_to_epoch):
        if not self.log_path.exists():
            return []
        try:
            old_df = pd.read_csv(self.log_path)
            old_df = old_df[old_df["Epoch"] <= up_to_epoch]
            print(f"History log restored up to epoch {up_to_epoch}.")
            return old_df.to_dict("records")
        except Exception as exc:
            print(f"Failed to restore history log: {exc}")
            return []

    def _restore_rng_state(self, rng_state):
        if not rng_state:
            return
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"].cpu())
        if torch.cuda.is_available() and rng_state.get("cuda"):
            torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["cuda"]])

    def try_resume_from_dir(self):
        if self.last_checkpoint_path.exists():
            checkpoint = self._load_torch_file(self.last_checkpoint_path)
            required = {
                "epoch", "model_state_dict", "optimizer_state_dict",
                "scheduler_state_dict", "best_val_mae", "best_epoch",
            }
            missing = sorted(required.difference(checkpoint))
            if missing:
                raise ValueError(f"Last checkpoint is missing keys: {missing}")

            completed_epoch = int(checkpoint["epoch"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if checkpoint.get("amp_scaler_state_dict"):
                self.grad_scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])
            self.best_val_mae = float(checkpoint["best_val_mae"])
            self.best_epoch = int(checkpoint["best_epoch"])
            best_model_name = checkpoint.get("best_model_name")
            if best_model_name:
                self.best_model_path = self.exp_dir / best_model_name
            if self.best_model_path is None or not self.best_model_path.exists():
                raise FileNotFoundError("Last checkpoint references a missing best model file")

            saved_logs = checkpoint.get("epoch_logs")
            self.epoch_logs = (
                list(saved_logs)
                if isinstance(saved_logs, list)
                else self._restore_history_from_csv(completed_epoch)
            )
            self._restore_rng_state(checkpoint.get("rng_state"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"Full-state resume: last_checkpoint.pth (completed epoch {completed_epoch}, "
                f"best epoch {self.best_epoch})"
            )
            return completed_epoch + 1

        model_files = glob.glob(str(self.exp_dir / "best_epoch*.pth"))
        if not model_files:
            print("No checkpoint found, starting from Epoch 1.")
            return 1

        def epoch_from_path(path):
            return int(Path(path).name.split("epoch", 1)[1].split("_", 1)[0])

        latest_model = max(model_files, key=epoch_from_path)
        filename = Path(latest_model).name
        start_epoch = epoch_from_path(latest_model)
        last_mae = float(filename.split("mae", 1)[1].replace(".pth", ""))
        print(
            f"Legacy weight-only resume: {filename} (epoch {start_epoch}); "
            "optimizer and scheduler start fresh."
        )
        self.model.load_state_dict(self._load_torch_file(latest_model))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.best_model_path = Path(latest_model)
        self.best_val_mae = last_mae
        self.best_epoch = start_epoch
        self.epoch_logs = self._restore_history_from_csv(start_epoch)
        return start_epoch + 1

    def _prepare_batch(self, batch):
        if isinstance(batch, dict):
            required = {"x", "x_mask", "y", "y_mask", "x_time", "y_time"}
            missing = sorted(required.difference(batch))
            if missing:
                raise ValueError(f"Context batch is missing keys: {missing}")
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            mask = batch["y_mask"].to(self.device, non_blocking=True)
            model_context = {
                "x_mask": batch["x_mask"].to(self.device, non_blocking=True),
                "x_time": batch["x_time"].to(self.device, non_blocking=True),
                "y_time": batch["y_time"].to(self.device, non_blocking=True),
            }
            return x, y, mask, model_context
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            x, y, mask = batch
            return (
                x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True),
                mask.to(self.device, non_blocking=True),
                {},
            )
        raise TypeError("Expected a three-tensor batch or a CARAMOE context dictionary")

    def _forward_model(self, x, model_context):
        output = self.model(x, **model_context)
        if isinstance(output, dict):
            if "prediction" not in output:
                raise ValueError("Model dictionary output does not contain 'prediction'")
            output = output["prediction"]
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Model output must be a tensor, got {type(output).__name__}")
        return output

    def _train_full_f1_batch(self, x, y, target_mask, model_context, batch_index):
        x_mask = model_context["x_mask"]
        augment_config = self.f1_training_config["augment"]
        if augment_config["enabled"]:
            augmented = self.f1_augmenter(x, x_mask)
            model_x = augmented.values
            training_context = dict(model_context)
            training_context["x_mask"] = augmented.mask
        else:
            augmented = None
            model_x = x
            training_context = model_context

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            output = self.model.forward_full(model_x, **training_context)
            loss, components = total_training_loss_v2(
                output,
                y,
                target_mask.bool(),
                expert_aux_weight=float(self.f1_training_config["expert_aux_weight"]),
                safe_weight=float(self.f1_training_config["safe_weight"]),
                high_reliability_threshold=float(
                    self.f1_training_config["high_reliability_threshold"]
                ),
                safe_margin=float(self.f1_training_config["safe_margin_normalized"]),
                training_variant=str(self.f1_training_config["training_variant"]),
                loss_mode=str(self.f1_training_config["loss_mode"]),
                pollutant_std=self.pollutant_std,
            )
            route_loss = loss.new_zeros(())
            if (
                augmented is not None
                and float(self.f1_training_config["route_consistency_weight"]) > 0
                and batch_index % int(self.f1_training_config["run_consistency_every"]) == 0
            ):
                with torch.no_grad():
                    original_output = self.model.forward_full(x, **model_context)
                route_loss = route_consistency_loss(
                    original_output["external_gate"].detach(),
                    output["external_gate"],
                    augmented.additionally_hidden_channels,
                )
                loss = loss + float(
                    self.f1_training_config["route_consistency_weight"]
                ) * route_loss
        return loss, components, route_loss

    def train_one_epoch(self, loader, epoch):
        if len(loader) == 0:
            raise ValueError("Training DataLoader is empty")
        self.model.train()
        totals = {
            "loss": 0.0,
            "forecast": 0.0,
            "forecast_aqb": 0.0,
            "forecast_balanced": 0.0,
            "expert_aux": 0.0,
            "safe": 0.0,
            "route": 0.0,
        }
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
        for batch_index, batch in enumerate(pbar):
            x, y, mask, model_context = self._prepare_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_full_f1_objective:
                loss, components, route_loss = self._train_full_f1_batch(
                    x, y, mask, model_context, batch_index
                )
            else:
                output = self._forward_model(x, model_context)
                if output.shape != y.shape:
                    raise ValueError(f"Model output shape {output.shape} != label shape {y.shape}")
                loss = self.metrics_tool._masked_mae(output, y, mask)
                components = {}
                route_loss = loss.detach().new_zeros(())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")

            self.grad_scaler.scale(loss).backward()
            if self.use_full_f1_objective and float(self.f1_training_config["grad_clip"]) > 0:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.f1_training_config["grad_clip"])
                )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            totals["loss"] += float(loss.detach())
            for name, value in components.items():
                totals[name] += float(value)
            totals["route"] += float(route_loss.detach())
            if self.use_full_f1_objective:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", forecast=f"{float(components['forecast']):.4f}"
                )
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        if self.use_full_f1_objective:
            return {name: value / len(loader) for name, value in totals.items()}
        return totals["loss"] / len(loader)

    @torch.no_grad()
    def eval_one_epoch(self, loader, mode="Validation"):
        if len(loader) == 0:
            raise ValueError(f"{mode} DataLoader is empty")
        self.model.eval()
        accumulator = self.metrics_tool.accumulator()
        pbar = tqdm(loader, desc=f"      [{mode}]", leave=False)
        for batch in pbar:
            x, y, mask, model_context = self._prepare_batch(batch)
            output = self._forward_model(x, model_context)
            if output.shape != y.shape:
                raise ValueError(f"Model output shape {output.shape} != label shape {y.shape}")
            accumulator.update(output, y, mask)
            running = accumulator.compute()
            pbar.set_postfix(MAE=f"{running['Global_MAE']:.2f}")
        metrics = accumulator.compute()
        if not np.isfinite(list(metrics.values())).all():
            raise FloatingPointError(f"{mode} produced non-finite metrics")
        return metrics

    def fit(self, train_loader, val_loader, test_loader):
        start_epoch = self.try_resume_from_dir()
        print(f"Experiment started | dir: {self.exp_dir}")
        if self.use_full_f1_objective:
            print(
                "CARAMOE objective: full aqb_mix + expert auxiliary + safe routing + "
                "route consistency + mask augmentation"
            )

        for epoch in range(start_epoch, self.args.epochs + 1):
            train_result = self.train_one_epoch(train_loader, epoch)
            if isinstance(train_result, dict):
                train_loss = train_result["loss"]
                train_components = train_result
            else:
                train_loss = float(train_result)
                train_components = {}
            val_metrics = self.eval_one_epoch(val_loader, mode="Validation")
            test_metrics = self.eval_one_epoch(test_loader, mode="Test")

            log_entry = {
                "Epoch": epoch,
                "Train_Loss": train_loss,
                "LR": self.optimizer.param_groups[0]["lr"],
            }
            log_entry.update(
                {f"Train_{key}": value for key, value in train_components.items() if key != "loss"}
            )
            log_entry.update({f"Val_{key}": value for key, value in val_metrics.items()})
            log_entry.update({f"Test_{key}": value for key, value in test_metrics.items()})
            self.epoch_logs.append(log_entry)
            pd.DataFrame(self.epoch_logs).to_csv(self.log_path, index=False)

            val_mae = val_metrics["Global_MAE"]
            print(
                f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
                f"Validation MAE: {val_mae:.2f} | "
                f"Test MAE: {test_metrics['Global_MAE']:.2f}"
            )
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_epoch = epoch
                self.save_best_model(epoch, val_mae)
            self.scheduler.step(val_mae)
            self.save_last_checkpoint(epoch)

        if self.best_model_path is None or not self.best_model_path.exists():
            raise FileNotFoundError("No best validation checkpoint is available for final testing")

        state_dict = self._load_torch_file(self.best_model_path)
        self.model.load_state_dict(state_dict)
        test_metrics = self.eval_one_epoch(test_loader, mode="Final Test")
        pd.DataFrame([test_metrics]).to_csv(self.final_test_path, index=False)

        summary = {
            "best_epoch": self.best_epoch,
            "best_validation_global_mae": self.best_val_mae,
            "best_model_path": str(self.best_model_path.resolve()),
            "metric_aggregation": "global numerator/denominator over valid mask entries",
            "uses_full_f1_training_objective": self.use_full_f1_objective,
            "final_test_metrics": test_metrics,
        }
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(
            f"Final test complete | best_epoch={self.best_epoch} | "
            f"Global MAE={test_metrics['Global_MAE']:.2f}"
        )
        return test_metrics

    def save_last_checkpoint(self, epoch):
        if self.best_model_path is None or not self.best_model_path.exists():
            raise FileNotFoundError("Cannot save last checkpoint without a best model")
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        payload = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "amp_scaler_state_dict": self.grad_scaler.state_dict(),
            "best_val_mae": float(self.best_val_mae),
            "best_epoch": int(self.best_epoch),
            "best_model_name": self.best_model_path.name,
            "epoch_logs": self.epoch_logs,
            "rng_state": rng_state,
        }
        temporary_path = Path(f"{self.last_checkpoint_path}.tmp")
        torch.save(payload, temporary_path)
        os.replace(temporary_path, self.last_checkpoint_path)

        metadata = {
            "epoch": int(epoch),
            "best_epoch": int(self.best_epoch),
            "best_validation_global_mae": float(self.best_val_mae),
        }
        temporary_meta_path = Path(f"{self.last_checkpoint_meta_path}.tmp")
        with temporary_meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_meta_path, self.last_checkpoint_meta_path)

    def save_best_model(self, epoch, mae):
        new_name = f"best_epoch{epoch}_mae{mae:.4f}.pth"
        new_path = self.exp_dir / new_name
        if self.best_model_path and self.best_model_path.exists():
            try:
                self.best_model_path.unlink()
            except OSError:
                pass
        torch.save(self.model.state_dict(), new_path)
        self.best_model_path = new_path
        print(f"Best model saved: {new_name}")
