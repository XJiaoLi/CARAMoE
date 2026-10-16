from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


from lib.paths import ROOT, METADATA, attachment_dir

from lib.config import ExperimentConfig
from lib.graph import load_or_build_graph
from model.ablation import CARAMOEAblation, LEAVE_ONE_OUT_GROUPS
from model.caramoe import CARAMOE
from engine.base import prepare_output_directory, seed_everything
from engine.residual import CARAMOETrainer

from lib.blair_common import (
    build_datasets,
    city_paths,
    export_predictions,
    file_sha256,
    load_manifest,
    load_scaler,
    make_loader,
)
from lib.blair_metrics import evaluate_artifact
from preprocess.blair import write_json


F1_FLAGS = {
    "loss_mode": "aqb_mix",
    "self_baseline": "last",
    "trust_floor": 0.1,
    "cross_mode": "lagged",
    "lag_hours": (0, 1, 3, 6, 12, 23),
    "adapter_rank": 8,
    "spatial_scales": (5, 10, 20),
}

SELF_ONLY_ABLATION = "c1_self_only"
SELF_ONLY_AUX_NORMALIZATION = 5.0
F1_ABLATION_MAP = {
    "full": "full",
    "self_only": SELF_ONLY_ABLATION,
    "uniform_routing": "c2_uniform_external",
    "static_routing": "c3_static_routing",
    "capability_only": "c6_capability_only",
    "neither_availability_nor_capability": "c7_neither_availability_nor_capability",
}


def resolve_f1_ablation(args: argparse.Namespace) -> tuple[str, str]:
    if bool(getattr(args, "self_only", False)):
        return "self_only", SELF_ONLY_ABLATION
    slug = str(getattr(args, "ablation", "full")).strip().lower().replace("-", "_")
    if slug not in F1_ABLATION_MAP:
        raise ValueError(
            f"Unknown F1 ablation {slug!r}; choose one of {sorted(F1_ABLATION_MAP)}"
        )
    return slug, F1_ABLATION_MAP[slug]


def build_experiment(args: argparse.Namespace):
    ablation_slug, canonical_ablation = resolve_f1_ablation(args)
    leave_one_out = (
        str(getattr(args, "leave_one_out", "none"))
        .strip()
        .lower()
        .replace("-", "_")
    )
    if leave_one_out != "none" and canonical_ablation != "full":
        raise ValueError("Expert leave-one-out requires --ablation full")
    self_only = ablation_slug == "self_only"
    controlled_ablation = canonical_ablation != "full" or leave_one_out != "none"
    paths = city_paths(args.processed_root, args.city)
    missing = [str(path) for key, path in paths.items() if key != "root" and not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run protocol.py first. Missing files: {missing}")
    manifest = load_manifest(paths["manifest"])
    mean, std = load_scaler(paths["scaler"])
    base_config = ROOT / "configs" / f"aqb_h{args.horizon}.yaml"
    config = ExperimentConfig.from_yaml(base_config)
    config.data.data_dir = str(paths["root"])
    config.data.metadata_path = str(paths["metadata"])
    config.data.scaler_path = str(paths["scaler"])
    config.data.capability_cache_path = str(paths["capability"])
    graph_k = max(F1_FLAGS["spatial_scales"])
    config.data.graph_cache_path = str(paths["root"] / f"caramoe_graph_k{graph_k}.npz")
    config.data.input_window = 24
    config.data.output_window = args.horizon
    config.data.train_step = args.horizon
    config.data.eval_step = args.horizon
    config.data.num_workers = args.num_workers
    config.model.knn_k = graph_k
    config.train.device = getattr(args, "device", "auto")
    config.train.seed = args.seed
    config.train.batch_size = args.batch_size
    config.train.output_dir = str(Path(args.output_dir).resolve())
    config.train.training_variant = "C"
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.max_train_batches is not None:
        config.train.max_train_batches = args.max_train_batches
    if args.max_eval_batches is not None:
        config.train.max_eval_batches = args.max_eval_batches
    if self_only:
        config.train.expert_aux_weight *= SELF_ONLY_AUX_NORMALIZATION
    config.validate()

    capability_archive = np.load(paths["capability"])
    coverage = capability_archive["coverage"]
    structural_capability = capability_archive["structural_capability"]
    neighbor_indices, distances_km, station_metadata = load_or_build_graph(
        paths["metadata"],
        config.data.graph_cache_path,
        k=graph_k,
    )
    model_class = CARAMOEAblation if controlled_ablation else CARAMOE
    model_kwargs = dict(
        coords_deg=station_metadata[["lat", "lon"]].to_numpy(np.float32),
        neighbor_indices=neighbor_indices,
        distances_km=distances_km,
        capability_coverage=coverage,
        capability_ever=structural_capability,
        output_horizon=args.horizon,
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
        self_baseline=F1_FLAGS["self_baseline"],
        trust_floor=F1_FLAGS["trust_floor"],
        cross_mode=F1_FLAGS["cross_mode"],
        lag_hours=F1_FLAGS["lag_hours"],
        adapter_rank=F1_FLAGS["adapter_rank"],
        spatial_scales=F1_FLAGS["spatial_scales"],
    )
    if controlled_ablation:
        model_kwargs["ablation"] = canonical_ablation
        model_kwargs["leave_one_out"] = leave_one_out
    model = model_class(**model_kwargs)
    datasets = build_datasets(paths, mean, std, args.horizon)
    loaders = {
        "train": make_loader(
            datasets["train"], config.train.batch_size, True, config.data.num_workers
        ),
        "val": make_loader(
            datasets["val"], args.eval_batch_size, False, config.data.num_workers
        ),
        "test": make_loader(
            datasets["test"], args.eval_batch_size, False, config.data.num_workers
        ),
    }
    identity = model.experiment_metadata() | {
        **F1_FLAGS,
        "lag_hours": list(F1_FLAGS["lag_hours"]),
        "spatial_scales": list(F1_FLAGS["spatial_scales"]),
        "loss_mode": F1_FLAGS["loss_mode"],
        "dataset": "KDD Cup 2018 Fresh Air (Zenodo missing version)",
        "city": args.city.casefold(),
        "horizon": args.horizon,
        "seed": args.seed,
        "train_batch_size": args.batch_size,
        "split_ratio": "3:1:1",
        "manifest_sha256": file_sha256(paths["manifest"]),
        "source_tsf_sha256": manifest["source_tsf_sha256"],
        "eval_batch_size": args.eval_batch_size,
        "test_evaluated_once": True,
    }
    if self_only:
        identity |= {
            "ablation": SELF_ONLY_ABLATION,
            "requested_expert_aux_weight": (
                config.train.expert_aux_weight / SELF_ONLY_AUX_NORMALIZATION
            ),
            "expert_aux_normalization_factor": SELF_ONLY_AUX_NORMALIZATION,
            "active_expert_count_for_aux": 1,
            "effective_expert_aux_weight": config.train.expert_aux_weight,
        }
    elif controlled_ablation:
        identity["ablation_slug"] = (
            f"loo_{leave_one_out}" if leave_one_out != "none" else ablation_slug
        )
    return config, model, loaders, datasets, mean, std, paths, identity


def main(argv: Iterable[str] | None = None) -> None:
    default_processed = attachment_dir("data_release") / "BLAir"
    parser = argparse.ArgumentParser(
        description="Train frozen F1-full or a controlled F1 ablation on one BLAIR city."
    )
    parser.add_argument("--city", required=True, choices=["beijing", "london"])
    parser.add_argument("--horizon", required=True, type=int, choices=[6, 12, 18, 24])
    parser.add_argument("--processed-root", default=str(default_processed))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", help="Runtime device: cpu, cuda:N, or auto.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    ablation_group = parser.add_mutually_exclusive_group()
    ablation_group.add_argument(
        "--self-only",
        action="store_true",
        help="Run the registered C1 ablation: prediction equals the residual self expert.",
    )
    ablation_group.add_argument(
        "--ablation",
        choices=tuple(F1_ABLATION_MAP),
        default="full",
        help="Run one controlled routing/context ablation; default is F1-full.",
    )
    parser.add_argument(
        "--leave-one-out",
        choices=tuple(LEAVE_ONE_OUT_GROUPS),
        default="none",
        help="Remove one external expert while retaining the F1-full router.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output_dir).resolve()
    if args.resume and (output_dir / "run_complete.json").exists():
        raise FileExistsError(f"Run is already complete: {output_dir}")
    config, model, loaders, datasets, mean, std, paths, identity = build_experiment(args)
    if args.dry_run:
        result = {
            "status": "dry_run",
            "identity": identity,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "windows": {split: len(dataset) for split, dataset in datasets.items()},
            "stations": model.num_stations,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    resume_checkpoint = prepare_output_directory(output_dir, resume=args.resume)
    seed_everything(config.train.seed)
    config, model, loaders, datasets, mean, std, paths, identity = build_experiment(args)
    trainer = CARAMOETrainer(
        model,
        config,
        mean,
        std,
        loss_mode=F1_FLAGS["loss_mode"],
        v2_metadata=identity,
        resume_checkpoint=resume_checkpoint,
        write_artifacts=True,
    )
    write_json(output_dir / "kdd_protocol_identity.json", identity)
    final = trainer.fit(loaders["train"], loaders["val"], None)
    trainer.model.eval()
    ablation_slug, _ = resolve_f1_ablation(args)
    if args.leave_one_out != "none":
        artifact_method = f"CARAMOE_loo_{args.leave_one_out}"
    else:
        artifact_method = (
            "CARAMOE"
            if ablation_slug == "full"
            else f"CARAMOE_{ablation_slug}"
        )
    artifact_path = export_predictions(
        loaders["test"],
        trainer.device,
        mean,
        std,
        lambda batch: trainer._forward(batch)["prediction"],
        output_dir / "test_predictions.npz",
        {
            "method": artifact_method,
            "model_class": type(model).__name__,
            "city": args.city.casefold(),
            "horizon": args.horizon,
            "seed": args.seed,
            "split": "test",
            "train_batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "station_ids": datasets["test"].station_ids,
            "pollutants": ["pm25", "pm10", "no2", "o3", "so2", "co"],
            "split_manifest_sha256": identity["manifest_sha256"],
            "source_tsf_sha256": identity["source_tsf_sha256"],
        },
        max_batches=args.max_test_batches,
    )
    test_metrics, _ = evaluate_artifact(artifact_path)
    final = identity | final | {f"test_{key}": value for key, value in test_metrics.items()}
    final["test_prediction_artifact"] = str(artifact_path)
    final["status"] = "complete"
    write_json(output_dir / "final_metrics.json", final)
    write_json(
        output_dir / "run_complete.json",
        {"status": "complete", "result": "final_metrics.json", "test_passes": 1},
    )
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
