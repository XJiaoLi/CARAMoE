from __future__ import annotations


import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import traceback
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lib.paths import ROOT, METADATA, attachment_dir
from lib.config import ExperimentConfig, checkpoint_config_fingerprint
from data_provider.airqualitybench import (
    AirQualityWindowDataset,
    choose_station_subset,
    load_or_compute_capability,
    load_scaler,
    make_dataloader,
)
from lib.graph import load_or_build_graph
from model.ablation import (
    F1_ABLATION_SPECS,
    LEAVE_ONE_OUT_GROUPS,
    CARAMOEAblation,
    canonical_f1_variant,
    resolve_excluded_experts,
)
from lib.origin_stats import export_origin_sufficient_statistics
from engine.base import prepare_output_directory, seed_everything
from engine.ablation import CARAMOEAblationTrainer


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return result


def _parse_expert_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: str | Path, *, hash_file: bool = True) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    record: dict[str, Any] = {
        "path": str(resolved),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if hash_file and resolved.is_file():
        record["sha256"] = _sha256(resolved)
    return record


def _git_record(cwd: Path) -> dict[str, Any]:
    def run(*command: str) -> str | None:
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    root = run("git", "rev-parse", "--show-toplevel")
    if root is None:
        return {"available": False}
    status = run("git", "status", "--porcelain")
    return {
        "available": True,
        "root": root,
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def _environment_record() -> dict[str, Any]:
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd().resolve()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def _parameter_inventory(model: torch.nn.Module) -> dict[str, Any]:
    modules: dict[str, dict[str, int]] = {}
    for name, parameter in model.named_parameters():
        top = name.split(".", 1)[0]
        entry = modules.setdefault(top, {"parameters": 0, "trainable_parameters": 0})
        entry["parameters"] += parameter.numel()
        if parameter.requires_grad:
            entry["trainable_parameters"] += parameter.numel()
    return {
        "total_parameters": sum(item.numel() for item in model.parameters()),
        "trainable_parameters": sum(
            item.numel() for item in model.parameters() if item.requires_grad
        ),
        "by_top_level_module": modules,
    }


def _data_manifest(
    config: ExperimentConfig, *, include_test_file_metadata: bool
) -> dict[str, Any]:
    base = Path(config.data.data_dir)
    years = set(config.data.train_years) | set(config.data.val_years)
    if include_test_file_metadata:
        years |= set(config.data.test_years)
    years = sorted(years)
    return {
        "train_years": list(config.data.train_years),
        "validation_years": list(config.data.val_years),
        "test_years": list(config.data.test_years),
        "test_file_metadata_inspected": bool(include_test_file_metadata),
        "year_files": [
            _file_record(base / f"aq_compact_{year}.h5", hash_file=False) for year in years
        ],
        "scaler": _file_record(config.data.scaler_path),
        "capability": _file_record(config.data.capability_cache_path),
        "metadata": _file_record(config.data.metadata_path),
        "graph_cache": _file_record(config.data.graph_cache_path),
    }


def build_experiment(config: ExperimentConfig, args: argparse.Namespace):
    mean, std = load_scaler(config.data.scaler_path)
    capability = load_or_compute_capability(
        config.data.data_dir,
        config.data.train_years,
        config.data.capability_cache_path,
    )
    subset = choose_station_subset(
        capability["coverage"], config.data.max_stations, config.data.station_subset_path
    )
    graph_k = max(args.spatial_scales)
    neighbor_indices, distances_km, metadata = load_or_build_graph(
        config.data.metadata_path,
        config.data.graph_cache_path,
        k=graph_k,
        station_indices=subset,
    )
    coverage = capability["coverage"] if subset is None else capability["coverage"][subset]
    ever = capability["ever_observed"] if subset is None else capability["ever_observed"][subset]

    common_dataset = dict(
        data_dir=config.data.data_dir,
        mean=mean,
        std=std,
        input_window=config.data.input_window,
        output_window=config.data.output_window,
        station_indices=subset,
        cross_year_windows=config.data.cross_year_windows,
    )
    datasets = {
        "train": AirQualityWindowDataset(
            years=config.data.train_years, step=config.data.train_step, **common_dataset
        ),
        "val": AirQualityWindowDataset(
            years=config.data.val_years, step=config.data.eval_step, **common_dataset
        ),
    }
    if not args.skip_test:
        datasets["test"] = AirQualityWindowDataset(
            years=config.data.test_years, step=config.data.eval_step, **common_dataset
        )

    model = CARAMOEAblation(
        coords_deg=metadata[["lat", "lon"]].to_numpy(np.float32),
        neighbor_indices=neighbor_indices,
        distances_km=distances_km,
        capability_coverage=coverage,
        capability_ever=ever,
        output_horizon=config.data.output_window,
        hidden_dim=config.model.hidden_dim,
        station_dim=config.model.station_dim,
        pollutant_dim=config.model.pollutant_dim,
        time_dim=config.model.time_dim,
        router_dim=config.model.router_dim,
        num_pollutants=config.model.num_pollutants,
        spatial_chunk_size=config.model.spatial_chunk_size,
        dropout=config.model.dropout,
        station_id_dropout=config.model.station_id_dropout,
        use_station_id=config.model.use_station_id,
        decoder_layers=config.model.decoder_layers,
        input_profile=config.model.input_profile,
        self_baseline=args.self_baseline,
        trust_floor=args.trust_floor,
        cross_mode=args.cross_mode,
        lag_hours=args.lag_hours,
        adapter_rank=args.adapter_rank,
        spatial_scales=args.spatial_scales,
        ablation=args.ablation,
        leave_one_out=args.leave_one_out,
        excluded_experts=args.excluded_experts,
        freshness_mode=args.freshness_mode,
    )
    loaders = {
        name: make_dataloader(
            dataset,
            batch_size=config.train.batch_size,
            shuffle=name == "train",
            num_workers=config.data.num_workers,
        )
        for name, dataset in datasets.items()
    }
    return model, loaders, mean, std, subset


def _artifact_hashes(output_dir: Path) -> dict[str, Any]:
    names = (
        "best.pt",
        "last.pt",
        "config.json",
        "history.csv",
        "model_variant.json",
        "model_summary.json",
        "parameter_inventory.json",
        "environment.json",
        "data_manifest.json",
        "station_subset.npy",
        "validation_metrics.json",
        "final_metrics.json",
        "run_manifest.json",
        "run_complete.json",
        "validation_origin_stats.npz",
        "validation_origin_stats.json",
    )
    return {
        name: _file_record(output_dir / name)
        for name in names
        if (output_dir / name).exists()
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train frozen F1-full or one controlled C1--C10/LOO ablation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", help="Runtime device: cpu or cuda:N; overrides config only.")
    parser.add_argument("--ablation", default="full")
    parser.add_argument(
        "--leave-one-out", choices=tuple(LEAVE_ONE_OUT_GROUPS), default="none"
    )
    parser.add_argument(
        "--excluded-experts", type=_parse_expert_tuple, default=()
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--loss-mode", choices=["current", "aqb_mix"], default="aqb_mix")
    parser.add_argument("--self-baseline", choices=["last", "last_daily"], default="last")
    parser.add_argument("--trust-floor", type=float, default=0.1)
    parser.add_argument("--cross-mode", choices=["current", "lagged"], default="lagged")
    parser.add_argument(
        "--lag-hours", type=_parse_int_tuple, default=(0, 1, 3, 6, 12, 23)
    )
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument(
        "--freshness-mode",
        choices=["decay", "binary"],
        default="decay",
        help="decay is frozen F1; binary keeps validity but removes gap-decay magnitude.",
    )
    parser.add_argument(
        "--spatial-scales", type=_parse_int_tuple, default=(5, 10, 20)
    )
    parser.add_argument("--graph-cache")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Defaults to 0 to avoid persistent-worker stalls observed on Windows.",
    )
    parser.add_argument(
        "--no-export-val-origin-stats",
        dest="export_val_origin_stats",
        action="store_false",
        help="Opt out of the default post-training 2024 per-origin sufficient-stat export.",
    )
    parser.set_defaults(export_val_origin_stats=True)
    return parser


def _normalize_args(args: argparse.Namespace) -> None:
    try:
        args.ablation = canonical_f1_variant(args.ablation)
        args.excluded_experts = resolve_excluded_experts(
            args.leave_one_out, args.excluded_experts
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    frozen_f1 = (
        args.ablation == "full"
        and args.leave_one_out == "none"
        and not args.excluded_experts
        and args.loss_mode == "aqb_mix"
        and args.self_baseline == "last"
        and args.trust_floor == 0.1
        and args.cross_mode == "lagged"
        and args.lag_hours == (0, 1, 3, 6, 12, 23)
        and args.adapter_rank == 8
        and args.spatial_scales == (5, 10, 20)
        and args.freshness_mode == "decay"
    )
    if not args.skip_test and not frozen_f1:
        raise SystemExit(
            "2025 test access is allowed only for the exact frozen F1-full specification. "
            "C1--C10, leave-one-out, and sensitivity runs must include --skip-test."
        )


def _guard_config_test_access(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if not args.skip_test and config.model.input_profile.lower() != "full":
        raise SystemExit("2025 test access is restricted to input_profile=full F1 runs.")


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    _normalize_args(args)

    config = ExperimentConfig.from_yaml(args.config)
    config.data.data_dir = str(attachment_dir("data_release") / "AirQualityBench")
    for key in ("metadata_path", "scaler_path", "graph_cache_path", "capability_cache_path"):
        name = Path(getattr(config.data, key)).name
        setattr(config.data, key, str(METADATA / "AirQualityBench" / name))
    _guard_config_test_access(config, args)
    if args.device is not None:
        config.train.device = args.device
    config.train.seed = args.seed
    config.train.output_dir = args.output_dir
    config.train.training_variant = "C"
    config.train.epochs = args.epochs
    if args.max_train_batches is not None:
        config.train.max_train_batches = args.max_train_batches
    if args.max_eval_batches is not None:
        config.train.max_eval_batches = args.max_eval_batches
    config.data.num_workers = args.num_workers

    spec = F1_ABLATION_SPECS[args.ablation]
    if spec.disable_safe_loss:
        config.train.safe_weight = 0.0
    if spec.disable_route_consistency:
        config.train.route_consistency_weight = 0.0
    requested_expert_aux_weight = float(config.train.expert_aux_weight)
    active_external = 0 if spec.router_mode == "self_only" else 4 - len(args.excluded_experts)
    active_expert_count = 1 + active_external
    expert_aux_normalization_factor = 5.0 / active_expert_count
    config.train.expert_aux_weight *= expert_aux_normalization_factor

    graph_k = max(args.spatial_scales)
    config.model.knn_k = graph_k
    if args.graph_cache is not None:
        config.data.graph_cache_path = args.graph_cache
    elif graph_k > 10:
        config.data.graph_cache_path = "./data/caramoe_graph_k20.npz"
    config.validate()

    output_dir = Path(config.train.output_dir)
    if args.resume and (output_dir / "run_complete.json").exists():
        raise FileExistsError(f"Run is already complete: {output_dir / 'run_complete.json'}")
    existing_final_path = output_dir / "final_metrics.json"
    if args.resume and existing_final_path.exists():
        with open(existing_final_path, "r", encoding="utf-8") as handle:
            existing_final = json.load(handle)
        if any(str(key).startswith("test_") for key in existing_final):
            _write_json_atomic(
                output_dir / "run_complete.json",
                {"status": "recovered_after_final_test", "result": existing_final_path.name},
            )
            print(json.dumps(existing_final, indent=2, ensure_ascii=False), flush=True)
            return

    resume_checkpoint = prepare_output_directory(output_dir, resume=args.resume)
    seed_everything(config.train.seed)
    model, loaders, mean, std, subset = build_experiment(config, args)
    metadata = model.experiment_metadata() | {
        "loss_mode": args.loss_mode,
        "seed": config.train.seed,
        "validation_only": bool(args.skip_test),
        "graph_k": graph_k,
        "graph_cache_path": config.data.graph_cache_path,
        "safe_weight": config.train.safe_weight,
        "route_consistency_weight": config.train.route_consistency_weight,
        "expert_aux_weight": config.train.expert_aux_weight,
        "requested_expert_aux_weight": requested_expert_aux_weight,
        "expert_aux_normalization_factor": expert_aux_normalization_factor,
        "active_expert_count_for_aux": active_expert_count,
        "config_fingerprint": checkpoint_config_fingerprint(config),
        "export_val_origin_stats": bool(args.export_val_origin_stats),
        "test_access_policy": (
            "validation_only_no_test_dataset" if args.skip_test else "one_final_test_evaluation"
        ),
    }

    identity_path = output_dir / "model_variant.json"
    if args.resume:
        with open(identity_path, "r", encoding="utf-8") as handle:
            saved_metadata = json.load(handle)
        if saved_metadata != metadata:
            raise ValueError(
                f"Cannot resume a different F1 experiment: saved={saved_metadata}, "
                f"requested={metadata}."
            )
    else:
        _write_json_atomic(identity_path, metadata)
        if subset is not None:
            np.save(output_dir / "station_subset.npy", subset)
        inventory = _parameter_inventory(model)
        _write_json_atomic(output_dir / "parameter_inventory.json", inventory)
        _write_json_atomic(
            output_dir / "model_summary.json",
            metadata
            | inventory
            | {
                "model_class": type(model).__name__,
                "stations": model.num_stations,
                "pollutants": model.num_pollutants,
                "input_window": config.data.input_window,
                "horizon": model.output_horizon,
                "device_requested": config.train.device,
            },
        )
        root = Path(__file__).resolve().parent
        source_files = [Path(__file__).resolve(), Path(args.config).resolve()]
        source_files.extend(sorted((root / "model").glob("*.py")))
        _write_json_atomic(output_dir / "environment.json", _environment_record())
        _write_json_atomic(
            output_dir / "data_manifest.json",
            _data_manifest(config, include_test_file_metadata=not args.skip_test),
        )
        _write_json_atomic(
            output_dir / "run_manifest.json",
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "identity": metadata,
                "resolved_config": config.to_dict(),
                "environment": _environment_record(),
                "git": _git_record(root),
                "source_files": [_file_record(path) for path in source_files],
                "data_manifest_file": "data_manifest.json",
                "parameter_inventory_file": "parameter_inventory.json",
            },
        )

    trainer = CARAMOEAblationTrainer(
        model,
        config,
        mean,
        std,
        loss_mode=args.loss_mode,
        v2_metadata=metadata,
        resume_checkpoint=resume_checkpoint,
        write_artifacts=True,
    )
    print(
        json.dumps(
            {
                "event": "run_start",
                "device": str(trainer.device),
                "cuda_name": (
                    torch.cuda.get_device_name(trainer.device)
                    if trainer.device.type == "cuda"
                    else None
                ),
                "metadata": metadata,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    run_started_utc = datetime.now(timezone.utc).isoformat()
    run_started_clock = time.perf_counter()
    try:
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
        final = trainer.fit(loaders["train"], loaders["val"], loaders.get("test"))
        if trainer.best_validation_metrics is None:
            raise RuntimeError("Best-validation metrics were not retained after restore.")
        origin_statistics = None
        if args.export_val_origin_stats:
            origin_statistics = export_origin_sufficient_statistics(
                trainer.model,
                loaders["val"],
                device=trainer.device,
                use_amp=trainer.use_amp,
                mean=mean,
                std=std,
                output_path=output_dir / "validation_origin_stats.npz",
                split="validation",
                years=list(config.data.val_years),
                max_batches=config.train.max_eval_batches,
                test_years_touched=False,
            )
        completed_utc = datetime.now(timezone.utc).isoformat()
        result = metadata | final | {
            "best_validation": trainer.best_validation_metrics,
            "peak_cuda_memory_mib": (
                torch.cuda.max_memory_allocated(trainer.device) / (1024.0**2)
                if trainer.device.type == "cuda"
                else 0.0
            ),
            "run_started_utc": run_started_utc,
            "completed_utc": completed_utc,
            "total_wall_seconds": time.perf_counter() - run_started_clock,
            "validation_years_touched": True,
            "test_years_touched": not args.skip_test,
            "test_dataset_constructed": not args.skip_test,
            "validation_origin_statistics": origin_statistics,
            "status": "complete",
        }
        result_path = output_dir / (
            "validation_metrics.json" if args.skip_test else "final_metrics.json"
        )
        _write_json_atomic(result_path, result)
        _write_json_atomic(
            output_dir / "run_complete.json",
            {
                "status": "complete",
                "result": result_path.name,
                "completed_utc": result["completed_utc"],
            },
        )
        _write_json_atomic(output_dir / "artifact_hashes.json", _artifact_hashes(output_dir))
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    except BaseException as exc:
        _write_json_atomic(
            output_dir / "run_failed.json",
            {
                "status": "failed",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "resume_checkpoint": "last.pt" if (output_dir / "last.pt").exists() else None,
            },
        )
        raise


if __name__ == "__main__":
    main()
