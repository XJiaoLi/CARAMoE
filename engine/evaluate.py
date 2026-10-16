from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.paths import ROOT, METADATA, attachment_dir
from run import (require_data, select_run,
                 trained_checkpoint, output_path, disk_path, checkpoint_path)

POLLUTANTS = ("pm25", "pm10", "no2", "o3", "so2", "co")


def read_json(path):
    return json.loads(disk_path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    digest = hashlib.sha256()
    with disk_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_fields(actual, expected, label):
    for key, value in expected.items():
        if key not in actual:
            raise ValueError(f"{label} lacks required field: {key}")
        if isinstance(value, dict):
            require_fields(actual[key], value, f"{label}.{key}")
        elif actual[key] != value:
            raise ValueError(f"{label}.{key} differs from the primary protocol: {actual[key]!r} != {value!r}")


def validate_training_sidecars(run, checkpoint):
    directory = checkpoint.parent
    common_identity = dict(self_baseline="last", trust_floor=0.1, cross_mode="lagged",
        lag_hours=[0, 1, 3, 6, 12, 23], adapter_rank=8, residual_self=True, temporal_layers=2)
    if run["dataset"] in ("aqb", "blair"):
        import yaml
        code_root = ROOT
        expected = yaml.safe_load((code_root / f"configs/aqb_h{run['horizon']}.yaml").read_text(encoding="utf-8"))
        actual = read_json(directory / "config.json")
        data_identity = {key: value for key, value in expected["data"].items()
                         if key not in ("data_dir", "metadata_path", "scaler_path", "graph_cache_path", "capability_cache_path", "num_workers")}
        model_identity = expected["model"] | {"knn_k": 20}
        train_identity = {key: value for key, value in expected["train"].items()
                          if key not in ("output_dir", "device", "log_every")}
        train_identity.update(seed=run["seed"], epochs=50, batch_size=2 if run["dataset"] == "aqb" else 8)
        require_fields(actual, dict(data=data_identity, model=model_identity, augment=expected["augment"], train=train_identity), "config.json")
        identity_name = "model_variant.json" if run["dataset"] == "aqb" else "kdd_protocol_identity.json"
        identity = read_json(directory / identity_name)
        require_fields(identity, common_identity | dict(spatial_scales=[5, 10, 20], loss_mode="aqb_mix", seed=run["seed"]), identity_name)
        if identity.get("model_architecture") not in ("CARAMOE", "CARAMOE_ablation"):
            raise ValueError("Checkpoint sidecar is not a supported full CARAMOE architecture")
        for key, expected_value in dict(ablation="full", leave_one_out="none", excluded_experts=[], freshness_mode="decay").items():
            if key in identity and identity[key] != expected_value:
                raise ValueError(f"{identity_name}.{key} is not the full primary model")
        if identity["model_architecture"] == "CARAMOE_ablation":
            require_fields(identity, dict(ablation="full", leave_one_out="none", excluded_experts=[], freshness_mode="decay"), identity_name)
        if run["dataset"] == "aqb":
            require_fields(identity, dict(graph_k=20, validation_only=False), identity_name)
        else:
            require_fields(identity, dict(city=run["city"], horizon=run["horizon"], train_batch_size=8,
                eval_batch_size=4, split_ratio="3:1:1"), identity_name)
        names = ("config.json", identity_name)
    else:
        expected = read_json(ROOT / f"configs/chinaair_k{run['graph_k']}.json")
        actual = read_json(directory / "effective_config.json")
        required = dict(dataset=expected["dataset"], model="CARAMOE", frequency="daily",
            train_years=expected["train_years"], val_years=expected["validation_years"], test_years=expected["test_years"],
            w_in=30, w_out=run["horizon"], step=run["horizon"], seed=run["seed"], epochs=50,
            batch_size=expected["batch_size_by_horizon"][str(run["horizon"])], lr=expected["learning_rate"],
            weight_decay=expected["weight_decay"], parameters=expected["parameters"],
            caramoe_config=expected["caramoe"], caramoe_training_config=expected["caramoe_training"],
            resolved_caramoe_training_config=expected["caramoe_training"], uses_full_f1_training_objective=True)
        require_fields(actual, required, "effective_config.json")
        identity = read_json(directory / "model_metadata.json")
        require_fields(identity, common_identity | dict(model_architecture="CARAMOE",
            spatial_scales=expected["caramoe"]["spatial_scales"], knn_k=run["graph_k"],
            frequency="daily", resolved_training_config=expected["caramoe_training"]), "model_metadata.json")
        names = ("effective_config.json", "model_metadata.json")
    return [dict(artifact=str(directory / name), sha256=file_sha256(directory / name)) for name in names]


def verified_checkpoint(run, user_checkpoint):
    path = trained_checkpoint(user_checkpoint)
    digest = file_sha256(path)
    sidecars = validate_training_sidecars(run, path)
    return path, digest, "user_trained", sidecars


def aqb_config_from_mapping(raw):
    from dataclasses import fields, replace
    from lib.config import ExperimentConfig
    base = ExperimentConfig.default()
    return replace(base, **{field.name: replace(getattr(base, field.name), **raw.get(field.name, {}))
                            for field in fields(base)})


def load_weights(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def iter_batches(loader, limit):
    import itertools
    return iter(loader) if limit is None else itertools.islice(loader, limit)


def evaluate_aqb(args, run, checkpoint, device):
    import torch
    from lib.checkpoint import _load_capability_readonly, _load_graph_readonly
    from data_provider.airqualitybench import AirQualityWindowDataset, load_scaler, make_dataloader
    from lib.checkpoint import model_from_metadata
    from lib.batch_metrics import LegacyAQBBatchAccumulator
    from lib.metrics import MetricAccumulator
    from engine.base import move_batch, seed_everything
    saved = load_weights(checkpoint)
    raw_config = read_json(checkpoint.parent / "config.json")
    config = aqb_config_from_mapping(raw_config)
    config.data.data_dir = str(attachment_dir("data_release") / "AirQualityBench")
    for key in ("metadata_path", "scaler_path", "capability_cache_path", "graph_cache_path"):
        name = str(getattr(config.data, key)).replace("\\", "/").split("/")[-1]
        setattr(config.data, key, str(METADATA / "AirQualityBench" / name))
    if config.data.max_stations is not None or config.data.station_subset_path is not None:
        raise ValueError("Main runs must use the complete station set")
    if config.data.output_window != run["horizon"] or config.train.batch_size != 2:
        raise ValueError("AQB main-run protocol does not match the registry")
    metadata = read_json(checkpoint.parent / "model_variant.json")
    mean, std = load_scaler(config.data.scaler_path)
    capability = _load_capability_readonly(Path(config.data.capability_cache_path))
    graph_k = int(metadata.get("graph_k", max(metadata["spatial_scales"])))
    neighbors, distances, coords, _ = _load_graph_readonly(config, graph_k, None)
    model = model_from_metadata(config, metadata, coords_deg=coords, neighbor_indices=neighbors,
        distances_km=distances, capability_coverage=capability["coverage"], capability_ever=capability["ever_observed"])
    model.load_state_dict(saved["model"], strict=True)
    seed_everything(run["seed"])
    model.to(device).eval()
    years = config.data.test_years if args.split == "test" else config.data.val_years
    dataset = AirQualityWindowDataset(data_dir=config.data.data_dir, years=years, mean=mean, std=std,
        input_window=config.data.input_window, output_window=config.data.output_window,
        step=config.data.eval_step, cross_year_windows=config.data.cross_year_windows)
    loader = make_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)
    exact, legacy = MetricAccumulator(mean, std), LegacyAQBBatchAccumulator()
    amp = args.precision == "amp" and device.type == "cuda"
    samples, batches = 0, 0
    with torch.inference_mode():
        for raw in iter_batches(loader, args.max_batches):
            batch = move_batch(raw, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                prediction = model(batch["x"], batch["x_mask"], batch["x_time"], batch["y_time"])["prediction"]
            exact.update(prediction, batch["y"], batch["y_mask"])
            legacy.update_tensors(prediction, batch["y"], batch["y_mask"], mean, std)
            samples += len(batch["x"])
            batches += 1
            if batches % 50 == 0:
                print(f"AQB {args.split}: {batches}/{len(loader)} batches", flush=True)
    metrics = exact.compute()
    for level, values in legacy.compute().items():
        for name, value in values.items():
            key = f"Legacy_AQB_{name}" if level == "overall" else f"Legacy_AQB_{level}_{name}"
            metrics[key] = value
    return metrics, dict(samples=samples, total_samples=len(dataset), batch_count=batches, batch_size=2,
                        years=list(years), amp=bool(amp), checkpoint_epoch=int(saved.get("epoch", -1)))


def evaluate_blair(args, run, checkpoint, device):
    import torch
    from engine.blair import build_experiment
    from lib.blair_metrics import ExactLegacyAccumulator
    from engine.base import move_batch, seed_everything
    saved = load_weights(checkpoint)
    identity = read_json(checkpoint.parent / "kdd_protocol_identity.json")
    build_args = argparse.Namespace(city=run["city"], horizon=run["horizon"],
        processed_root=str(attachment_dir("data_release") / "BLAir"), output_dir=str(args.output_dir),
        seed=run["seed"], epochs=50, batch_size=8, eval_batch_size=4, num_workers=0,
        max_train_batches=None, max_eval_batches=None, device=str(device), ablation="full", leave_one_out="none")
    config, model, loaders, datasets, mean, std, paths, current_identity = build_experiment(build_args)
    for key in ("city", "horizon", "seed", "train_batch_size", "eval_batch_size", "manifest_sha256", "source_tsf_sha256"):
        if identity[key] != current_identity[key]:
            raise ValueError(f"BLAir protocol identity mismatch: {key}")
    model.load_state_dict(saved["model"], strict=True)
    seed_everything(run["seed"])
    model.to(device).eval()
    loader = loaders[args.split]
    accumulator = ExactLegacyAccumulator()
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, 1, 1, 6)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, 1, 1, 6)
    amp = args.precision == "amp" and device.type == "cuda"
    samples, batches = 0, 0
    with torch.inference_mode():
        for raw in iter_batches(loader, args.max_batches):
            batch = move_batch(raw, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                prediction = model(batch["x"], batch["x_mask"], batch["x_time"], batch["y_time"])["prediction"]
            physical_prediction = prediction.float() * (std_t + 1e-5) + mean_t
            physical_target = batch["y"].float() * (std_t + 1e-5) + mean_t
            accumulator.update(physical_prediction.cpu().numpy(), physical_target.cpu().numpy(), batch["y_mask"].cpu().numpy())
            samples += len(batch["x"])
            batches += 1
            if batches % 50 == 0:
                print(f"BLAir {args.split}: {batches}/{len(loader)} batches", flush=True)
    metrics = accumulator.compute()
    for name in ("MAE", "MSE", "RMSE", "MAPE"):
        metrics[f"Legacy_AQB_{name}"] = metrics[f"Legacy_AQB_Global_{name}"]
    return metrics, dict(samples=samples, total_samples=len(datasets[args.split]), batch_count=batches,
                        batch_size=4, amp=bool(amp), checkpoint_epoch=int(saved.get("epoch", -1)),
                        split_manifest_sha256=identity["manifest_sha256"])


def evaluate_china(args, run, checkpoint, device):
    import torch
    from data_provider.chinaair import GlobalAQDataset
    from model.chinaair import CARAMOEChinaAir
    from lib.china_dual_metrics import DualProtocolMaskedMetrics
    import pandas as pd
    config = read_json(checkpoint.parent / "effective_config.json")
    if (config["w_out"] != run["horizon"] or config["seed"] != run["seed"]
            or config["caramoe_config"]["knn_k"] != run["graph_k"]):
        raise ValueError("ChinaAir checkpoint identity does not match registry")
    data_dir = attachment_dir("data_release") / "ChinaAir"
    frame = pd.read_csv(METADATA / "ChinaAir/scaler.csv")
    if frame["parameter"].tolist() != list(POLLUTANTS):
        raise ValueError("ChinaAir scaler pollutant order mismatch")
    scaler = {"mean": frame["mean"].tolist(), "std": frame["std"].tolist()}
    common = dict(data_dir=str(data_dir), scaler_dict=scaler, adj_path=str(data_dir / "adjacency.pkl"),
        window_in=config["w_in"], window_out=config["w_out"], step=config["step"],
        expected_params=config["parameters"], expected_frequency="daily", return_context=True,
        mask_overlay_dir=str(data_dir / "masks"))
    train_set = GlobalAQDataset(years=config["train_years"], split_name="train", **common)
    years = config["test_years"] if args.split == "test" else config["val_years"]
    dataset = GlobalAQDataset(years=years, split_name="test" if args.split == "test" else "validation", **common)
    if train_set.station_ids != dataset.station_ids or train_set.params != dataset.params:
        raise ValueError("ChinaAir split station/pollutant identity mismatch")
    model = CARAMOEChinaAir.from_training_dataset(train_set, str(METADATA / "ChinaAir/selected_nodes_metadata.csv"),
        run["horizon"], config["caramoe_config"])
    model.load_state_dict(load_weights(checkpoint), strict=True)
    del train_set
    model.to(device).eval()
    batch_size = config["batch_size"]
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    accumulator = DualProtocolMaskedMetrics(scaler, device, params=list(POLLUTANTS)).accumulator()
    samples, batches = 0, 0
    with torch.inference_mode():
        for raw in iter_batches(loader, args.max_batches):
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw.items()}
            prediction = model(batch["x"], x_mask=batch["x_mask"], x_time=batch["x_time"], y_time=batch["y_time"])
            accumulator.update(prediction, batch["y"], batch["y_mask"])
            samples += len(batch["x"])
            batches += 1
    metrics = accumulator.compute()
    for pollutant in POLLUTANTS:
        for name in ("MAE", "MSE", "RMSE", "MAPE"):
            metrics[f"Legacy_AQB_{pollutant}_{name}"] = metrics[f"{pollutant}_Legacy_AQB_{name}"]
    return metrics, dict(samples=samples, total_samples=len(dataset), batch_count=batches,
                        batch_size=batch_size, years=years, amp=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["aqb", "blair", "chinaair"])
    parser.add_argument("--city", choices=["beijing", "london"])
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--graph-k", type=int, choices=[5, 10], help="ChinaAir only; must match the configured primary graph for the horizon")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Your trained weight file or training output directory")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--precision", choices=["native", "fp32", "amp"], default="native",
                        help="native/FP32 matches original evaluation; AMP is an optional AQB CUDA post-hoc audit")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args(argv)
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    run = select_run(args.dataset, args.city, args.horizon, args.graph_k)
    out = output_path(args.output_dir, args.checkpoint)
    require_data(run)
    checkpoint, checkpoint_sha256, checkpoint_source, checkpoint_sidecars = verified_checkpoint(run, args.checkpoint)
    args.checkpoint_source = checkpoint_source
    import torch
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Only cpu and cuda:N are supported")
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise RuntimeError(f"Requested device is unavailable: {device}")
    if args.precision == "amp" and (args.dataset != "aqb" or device.type != "cuda"):
        parser.error("--precision amp is only for the historical AQB CUDA post-hoc audit")
    started = time.perf_counter()
    evaluator = {"aqb": evaluate_aqb, "blair": evaluate_blair, "chinaair": evaluate_china}[args.dataset]
    metrics, details = evaluator(args, run, checkpoint, device)
    if not details["batch_count"]:
        raise RuntimeError("Evaluation loader is empty")
    if any(not math.isfinite(float(metrics[key])) for key in ("Global_MAE", "Global_RMSE", "Global_MAPE")):
        raise FloatingPointError("Evaluation produced non-finite global metrics")
    if args.max_batches is None and details["samples"] != details["total_samples"]:
        raise RuntimeError("Full evaluation did not visit every sample")
    result = dict(schema_version=1, status="partial_smoke" if args.max_batches is not None else "complete",
        model="CARAMOE", dataset=args.dataset, city=args.city, horizon=args.horizon, seed=run["seed"], split=args.split, graph_k=run["graph_k"],
        primary_mapping=run["primary"],
        device=str(device), precision=args.precision, effective_precision="amp" if details["amp"] else "fp32",
        max_batches=args.max_batches, **details,
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256, checkpoint_source=checkpoint_source, checkpoint_sidecars=checkpoint_sidecars,
        created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
        raw_metrics=metrics, **{f"{args.split}_{key}": value for key,value in metrics.items()},
        definitions={"Global": "Ratio of sums over observed target cells; dataset-native inverse transform and MAPE",
            "Legacy_AQB": "Equal-weight mean of deterministic per-batch metrics; RMSE inside batch; MAPE abs(error)/(abs(target)+0.1)",
            "MAPE_units": "ratio, not percent",
            "ChinaAir_Global_MAPE": "abs(error)/abs(target), excluding abs(target)<0.1; see Global_MAPE_Count",
            "precision": "Native evaluation is FP32 for all datasets; AMP is training-only in the original trainers. Optional AQB --precision amp matches the historical separate post-hoc recomputation path. Device/precision changes may change numbers."})
    out.mkdir(parents=True, exist_ok=True)
    target = out / "evaluation_metrics.json"
    target.write_text(json.dumps(clean_json(result), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=result["status"], output=str(target), samples=details["samples"],
                         Global_MAE=metrics["Global_MAE"], Legacy_AQB_MAE=metrics["Legacy_AQB_MAE"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
