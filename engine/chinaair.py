import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from lib.paths import ROOT, artifact_path
PROJECT_DIR = ROOT
DEFAULT_CONFIG_PATH = ROOT / "configs/chinaair_k5.json"

from data_provider.chinaair import GlobalAQDataset
from model.chinaair import CARAMOEChinaAir
from engine.china_trainer import Trainer


MODEL_CHOICES = ["CARAMOE"]


def load_config(config_path):
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "data_dir", "scaler_path", "adj_path", "metadata_path", "output_dir", "parameters",
        "train_years", "validation_years", "test_years", "window_in",
        "forecast_horizons", "step",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    return config


def resolve_project_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = artifact_path(path)
    return str(path.resolve())


def _resume_config_is_compatible(saved_config, args):
    value_keys = [
        "dataset", "frequency", "model", "train_years", "val_years", "test_years",
        "w_in", "w_out", "step", "parameters", "seed", "batch_size",
        "num_workers", "lr", "weight_decay",
    ]
    if args.model == "CARAMOE":
        value_keys.append("caramoe_config")
    for key in value_keys:
        if key not in saved_config or saved_config[key] != getattr(args, key):
            return False

    path_keys = ["data_dir", "scaler_path", "adj_path", "metadata_path"]
    for key in path_keys:
        if key not in saved_config:
            return False
        saved_path = str(Path(saved_config[key]).expanduser().resolve()).casefold()
        current_path = str(Path(getattr(args, key)).expanduser().resolve()).casefold()
        if saved_path != current_path:
            return False
    saved_overlay = saved_config.get("mask_overlay_dir", "")
    current_overlay = getattr(args, "mask_overlay_dir", "")
    if bool(saved_overlay) != bool(current_overlay):
        return False
    if current_overlay:
        saved_path = str(Path(saved_overlay).expanduser().resolve()).casefold()
        current_path = str(Path(current_overlay).expanduser().resolve()).casefold()
        if saved_path != current_path:
            return False
    return True


def _completed_epoch(exp_dir):
    last_checkpoint = exp_dir / "last_checkpoint.pth"
    last_metadata = exp_dir / "last_checkpoint.json"
    if last_checkpoint.exists() and last_metadata.exists():
        try:
            with last_metadata.open("r", encoding="utf-8") as handle:
                epoch = int(json.load(handle)["epoch"])
            if epoch > 0:
                return epoch
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    results_path = exp_dir / "results.csv"
    if last_checkpoint.exists() and results_path.exists():
        try:
            history = pd.read_csv(results_path, usecols=["Epoch"])
            if not history.empty:
                return int(history["Epoch"].max())
        except (OSError, KeyError, TypeError, ValueError):
            pass

    legacy_epochs = []
    for checkpoint in exp_dir.glob("best_epoch*_mae*.pth"):
        try:
            legacy_epochs.append(int(checkpoint.name.split("epoch", 1)[1].split("_", 1)[0]))
        except (IndexError, ValueError):
            continue
    return max(legacy_epochs) if legacy_epochs else None


def find_auto_resume_dir(args):
    output_root = Path(args.output_dir)
    if not output_root.exists():
        raise FileNotFoundError(
            f"Cannot auto-resume because output directory does not exist: {output_root}"
        )

    prefix = (
        f"{args.dataset}_{args.model}_{args.frequency}_"
        f"in{args.w_in}_out{args.w_out}_step{args.step}_"
    )
    candidates = []
    for exp_dir in output_root.glob(f"{prefix}*"):
        if not exp_dir.is_dir():
            continue
        config_path = exp_dir / "effective_config.json"
        if not config_path.exists():
            continue
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                saved_config = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not _resume_config_is_compatible(saved_config, args):
            continue
        epoch = _completed_epoch(exp_dir)
        if epoch is None:
            continue
        candidates.append((epoch, exp_dir.stat().st_mtime, exp_dir))

    if not candidates:
        raise FileNotFoundError(
            "No compatible resumable experiment was found under "
            f"{output_root} for model={args.model}, w_out={args.w_out}, step={args.step}."
        )

    epoch, _, selected = max(candidates, key=lambda item: (item[0], item[1]))
    print(f"Auto-resume selected | epoch={epoch} | dir={selected.resolve()}")
    return str(selected.resolve())


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    known, _ = config_parser.parse_known_args()
    config = load_config(known.config)

    parser = argparse.ArgumentParser(
        description="Train the full CARAMOE model on the daily ChinaAir dataset",
        parents=[config_parser],
    )
    parser.add_argument("--data_dir", type=str, default=config["data_dir"])
    parser.add_argument("--scaler_path", type=str, default=config["scaler_path"])
    parser.add_argument("--adj_path", type=str, default=config["adj_path"])
    parser.add_argument("--metadata_path", type=str, default=config["metadata_path"])
    parser.add_argument(
        "--mask_overlay_dir",
        type=str,
        default=config.get("mask_overlay_dir", ""),
        help="Optional directory containing mask_overlay_<year>.npz input masks",
    )
    parser.add_argument("--output_dir", type=str, default=config["output_dir"])
    parser.add_argument("--model", type=str, default=config.get("model", "CARAMOE"), choices=MODEL_CHOICES)
    parser.add_argument("--train_years", nargs="+", type=int, default=config["train_years"])
    parser.add_argument("--val_years", nargs="+", type=int, default=config["validation_years"])
    parser.add_argument("--test_years", nargs="+", type=int, default=config["test_years"])
    parser.add_argument("--step", type=int, default=config["step"])
    parser.add_argument("--w_in", type=int, default=config["window_in"])
    parser.add_argument(
        "--w_out",
        type=int,
        default=config["forecast_horizons"][0],
        choices=config["forecast_horizons"],
        help="Forecast horizon in days",
    )
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 20))
    parser.add_argument("--lr", type=float, default=config.get("learning_rate", 0.001))
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 50))
    parser.add_argument("--gpu", type=int, default=config.get("gpu", 0))
    parser.add_argument("--weight_decay", type=float, default=config.get("weight_decay", 1e-5))
    parser.add_argument("--num_workers", type=int, default=config.get("num_workers", 0))
    parser.add_argument("--seed", type=int, default=config.get("seed", 2026))
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="Automatically resume the most advanced compatible experiment",
    )
    resume_group.add_argument(
        "--resume_dir",
        type=str,
        default="",
        help="Manually select an experiment directory (legacy/advanced use)",
    )
    args = parser.parse_args()

    args.config = str(Path(args.config).expanduser().resolve())
    args.data_dir = resolve_project_path(args.data_dir)
    args.scaler_path = resolve_project_path(args.scaler_path)
    args.adj_path = resolve_project_path(args.adj_path)
    args.metadata_path = resolve_project_path(args.metadata_path)
    args.mask_overlay_dir = (
        resolve_project_path(args.mask_overlay_dir) if args.mask_overlay_dir else ""
    )
    args.output_dir = resolve_project_path(args.output_dir)
    args.resume_dir = resolve_project_path(args.resume_dir) if args.resume_dir else ""
    args.parameters = list(config["parameters"])
    args.frequency = config.get("frequency", "daily")
    args.dataset = config.get("dataset", "CHINA3238")
    args.allowed_horizons = list(config["forecast_horizons"])
    args.caramoe_config = dict(config.get("caramoe", {}))
    validate_args(args)
    if args.resume:
        args.resume_dir = find_auto_resume_dir(args)
    return args


def validate_args(args):
    if args.w_in <= 0 or args.w_out <= 0 or args.step <= 0:
        raise ValueError("w_in, w_out, and step must all be positive")
    if args.w_out not in args.allowed_horizons:
        raise ValueError(f"w_out must be one of {args.allowed_horizons}")
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("batch_size and epochs must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if args.model == "CARAMOE":
        lag_steps = [int(value) for value in args.caramoe_config.get(
            "lag_steps", [0, 1, 3, 6, 12, 23]
        )]
        if not lag_steps or min(lag_steps) < 0 or len(lag_steps) != len(set(lag_steps)):
            raise ValueError("caramoe lag_steps must be unique non-negative integers")
        if max(lag_steps) >= args.w_in:
            raise ValueError("caramoe maximum lag step must be smaller than w_in")
    split_years = args.train_years + args.val_years + args.test_years
    if len(split_years) != len(set(split_years)):
        raise ValueError("Train, validation, and test years must not overlap")
    if max(args.train_years) >= min(args.val_years) or max(args.val_years) >= min(args.test_years):
        raise ValueError("Expected chronological train < validation < test year splits")


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(gpu):
    if gpu == -1:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU index {gpu} is invalid; available devices: {torch.cuda.device_count()}")
    return torch.device(f"cuda:{gpu}")


def load_scaler(path, expected_parameters):
    scaler_path = Path(path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = pd.read_csv(scaler_path, encoding="utf-8-sig")
    required_columns = {"parameter", "mean", "std"}
    if not required_columns.issubset(scaler.columns):
        raise ValueError(f"Scaler must contain columns {sorted(required_columns)}")
    parameters = scaler["parameter"].astype(str).tolist()
    if parameters != expected_parameters:
        raise ValueError(f"Scaler parameter order {parameters} != expected {expected_parameters}")
    means = scaler["mean"].to_numpy(dtype=np.float32)
    stds = scaler["std"].to_numpy(dtype=np.float32)
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise ValueError("Scaler means/stds must be finite and every std must be positive")
    return {"mean": means.tolist(), "std": stds.tolist()}


def build_model(model_name, num_nodes, num_features, args, adj_mx, train_dataset=None):
    if model_name == "CARAMOE":
        if train_dataset is None:
            raise ValueError("caramoe requires the training dataset for causal capability priors")
        if num_features != 6:
            raise ValueError(f"caramoe requires six pollutant channels, got {num_features}")
        return CARAMOEChinaAir.from_training_dataset(
            train_dataset,
            metadata_path=args.metadata_path,
            output_horizon=args.w_out,
            model_config=args.caramoe_config,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def make_loader(dataset, args, shuffle, device):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def main(args):
    set_reproducible_seed(args.seed)
    device = select_device(args.gpu)
    scaler_dict = load_scaler(args.scaler_path, args.parameters)

    dataset_kwargs = {
        "data_dir": args.data_dir,
        "scaler_dict": scaler_dict,
        "adj_path": args.adj_path,
        "window_in": args.w_in,
        "window_out": args.w_out,
        "step": args.step,
        "expected_params": args.parameters,
        "expected_frequency": args.frequency,
        "return_context": args.model == "CARAMOE",
        "mask_overlay_dir": args.mask_overlay_dir,
    }
    train_set = GlobalAQDataset(years=args.train_years, split_name="train", **dataset_kwargs)
    val_set = GlobalAQDataset(years=args.val_years, split_name="validation", **dataset_kwargs)
    test_set = GlobalAQDataset(years=args.test_years, split_name="test", **dataset_kwargs)

    for split_set in (val_set, test_set):
        if split_set.station_ids != train_set.station_ids:
            raise ValueError(f"{split_set.split_name} station order does not match training data")
        if split_set.params != train_set.params:
            raise ValueError(f"{split_set.split_name} parameter order does not match training data")

    train_loader = make_loader(train_set, args, shuffle=True, device=device)
    val_loader = make_loader(val_set, args, shuffle=False, device=device)
    test_loader = make_loader(test_set, args, shuffle=False, device=device)

    model = build_model(
        args.model,
        num_nodes=train_set.num_nodes,
        num_features=train_set.num_features,
        args=args,
        adj_mx=train_set.get_adj(),
        train_dataset=train_set,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Experiment config | dataset={args.dataset} | frequency={args.frequency} | "
        f"device={device} | model={args.model} | nodes={train_set.num_nodes} | "
        f"features={train_set.num_features} | window={args.w_in}->{args.w_out} days | "
        f"batch_size={args.batch_size} | parameters={parameter_count:,}"
    )

    trainer = Trainer(model, scaler_dict, args, device, params=args.parameters)
    trainer.fit(train_loader, val_loader, test_loader)


REQUIRED_OBJECTIVE = {
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
}


def _load_and_validate_full_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("model") != "CARAMOE":
        raise ValueError("The full-F1 entrypoint requires config model='CARAMOE'")
    training = dict(config.get("caramoe_training", {}))
    missing = sorted(set(REQUIRED_OBJECTIVE).difference(training))
    if missing:
        raise ValueError(f"caramoe_training is missing frozen keys: {missing}")
    for key, expected in REQUIRED_OBJECTIVE.items():
        if training[key] != expected:
            raise ValueError(
                f"Frozen F1 objective mismatch for {key}: "
                f"expected {expected!r}, got {training[key]!r}"
            )
    augment = dict(training.get("augment", {}))
    if not augment.get("enabled", False):
        raise ValueError("Full F1 requires observation-mask augmentation")
    return config, training


def parse_full_f1_args():
    args = parse_args()
    if args.model != "CARAMOE":
        raise ValueError("This training entry point only accepts --model CARAMOE")
    config, training = _load_and_validate_full_config(args.config)
    args.caramoe_training_config = training

    knn_k = int(args.caramoe_config["knn_k"])
    spatial_scales = [int(value) for value in args.caramoe_config["spatial_scales"]]
    if knn_k not in {5, 10}:
        raise ValueError(f"Long-run CARAMOE graph must be K5 or K10, got K{knn_k}")
    if not spatial_scales or max(spatial_scales) > knn_k:
        raise ValueError("spatial_scales must be non-empty and cannot exceed knn_k")

    schedule = {int(key): int(value) for key, value in config["batch_size_by_horizon"].items()}
    expected_batch = schedule[args.w_out]
    if args.batch_size != expected_batch:
        raise ValueError(
            f"H{args.w_out} requires batch_size={expected_batch}, got {args.batch_size}"
        )
    return args


def inject_default_config(argv):
    arguments = list(argv)
    if "--config" not in arguments:
        arguments[1:1] = ["--config", str(DEFAULT_CONFIG_PATH)]
    return arguments


if __name__ == "__main__":
    sys.argv = inject_default_config(sys.argv)
    main(parse_full_f1_args())
