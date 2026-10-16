from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from lib.paths import ROOT, METADATA, attachment_dir, artifact_path


def disk_path(path):
    path = Path(path).resolve()
    value = str(path)
    if os.name == "nt" and len(value) >= 248 and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def manifest():
    return json.loads((ROOT / "configs/assets.json").read_text(encoding="utf-8"))["entries"]


def configured_runs():
    result = []
    for dataset, cities, horizons, seed in (
        ("aqb", (None,), (6, 12, 18, 24), 42),
        ("blair", ("beijing", "london"), (6, 12, 18, 24), 42),
        ("chinaair", (None,), (1, 3, 7, 14), 2026),
    ):
        for city in cities:
            for horizon in horizons:
                graph_k = (10 if horizon == 1 else 5) if dataset == "chinaair" else 20
                result.append(dict(dataset=dataset, city=city, horizon=horizon,
                                   seed=seed, graph_k=graph_k, primary=True))
    return result


def select_run(dataset, city, horizon, graph_k=None):
    if graph_k is not None and dataset != "chinaair":
        raise ValueError("--graph-k is a ChinaAir primary-configuration check")
    runs = [run for run in configured_runs() if run["dataset"] == dataset
            and run["horizon"] == horizon and run["city"] == city
            and (graph_k is None or run["graph_k"] == graph_k)]
    if len(runs) != 1:
        raise ValueError("Choose one registered run: aqb H6/12/18/24; blair plus city H6/12/18/24; chinaair H1/3/7/14.")
    return runs[0]


def require_data(run):
    inputs = [e for e in manifest() if e["artifact"].startswith("data_release/")
              and e["dataset"] == run["dataset"]]
    if run["dataset"] == "blair":
        inputs = [e for e in inputs if Path(e["artifact"]).stem == run["city"]]
    missing = [e["artifact"] for e in inputs if not artifact_path(e["artifact"]).is_file()]
    if missing:
        raise FileNotFoundError("Place the data_release directory beside run.py. Missing:\n" + "\n".join(missing))


def checkpoint_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def trained_checkpoint(value):
    candidate = checkpoint_path(value)
    if candidate.is_file():
        if candidate.suffix.lower() not in (".pt", ".pth"):
            raise ValueError("Checkpoint must be a .pt or .pth training output")
        return disk_path(candidate)
    if not candidate.is_dir():
        raise FileNotFoundError(f"Training output not found: {candidate}. Run train first.")
    weights = sorted(set(candidate.rglob("best.pt")) | set(candidate.rglob("best_epoch*_mae*.pth")))
    if not weights:
        raise FileNotFoundError(f"No best checkpoint found in {candidate}. Complete training first.")
    if len(weights) != 1:
        raise ValueError(f"Multiple best checkpoints found in {candidate}; select one experiment directory or weight file")
    return disk_path(weights[0])


def output_path(value, input_checkpoint=None):
    path = Path(value).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if path == ROOT or path in ROOT.parents:
        raise ValueError("Output directory must be a new experiment subdirectory")
    if input_checkpoint is not None:
        selected = checkpoint_path(input_checkpoint)
        source = selected if selected.is_dir() or not selected.suffix else selected.parent
        if path == source or source in path.parents or path in source.parents:
            raise ValueError("Output overlaps the input checkpoint directory")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {path}")
    return path


def train_command(run, device, out):
    h = str(run["horizon"])
    py = [sys.executable, "-B", "-m"]
    if run["dataset"] == "aqb":
        return ROOT, py + ["engine.airqualitybench", "--config", str(ROOT / f"configs/aqb_h{h}.yaml"),
            "--output-dir", str(out), "--device", device, "--seed", "42", "--epochs", "50",
            "--ablation", "full", "--loss-mode", "aqb_mix", "--self-baseline", "last",
            "--trust-floor", "0.1", "--cross-mode", "lagged", "--lag-hours", "0,1,3,6,12,23",
            "--adapter-rank", "8", "--spatial-scales", "5,10,20", "--graph-cache",
            str(METADATA / "AirQualityBench/caramoe_graph_k20.npz"), "--num-workers", "0"]
    if run["dataset"] == "blair":
        return ROOT, py + ["engine.blair", "--city", run["city"], "--horizon", h,
            "--seed", "42", "--epochs", "50", "--batch-size", "8", "--eval-batch-size", "4",
            "--num-workers", "0", "--device", device, "--processed-root",
            str(attachment_dir("data_release") / "BLAir"), "--output-dir", str(out)]
    gpu = "-1" if device == "cpu" else device.split(":")[-1]
    return ROOT, py + ["engine.chinaair", "--config",
        str(ROOT / f"configs/chinaair_k{run['graph_k']}.json"), "--model", "CARAMOE",
        "--w_in", "30", "--w_out", h, "--step", h,
        "--batch_size", str({1:20,3:8,7:4,14:2}[run["horizon"]]),
        "--epochs", "50", "--seed", "2026", "--num_workers", "0", "--gpu", gpu,
        "--output_dir", str(out)]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "predict", "list", "evaluate"])
    parser.add_argument("--dataset", choices=["aqb", "blair", "chinaair"])
    parser.add_argument("--city", choices=["beijing", "london"])
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--graph-k", type=int, choices=[5, 10], help="ChinaAir only: assert the registered graph size (H1=10, H3/H7/H14=5)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N (explicit; no automatic fallback)")
    parser.add_argument("--output-dir", help="New output directory; relative paths are repository-relative")
    parser.add_argument("--checkpoint", help="Prediction only: your trained weight file or training output directory; required")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--precision", choices=["native", "fp32", "amp"], default="native",
                        help="Prediction only. native is FP32; amp is an optional AQB CUDA post-hoc audit.")
    parser.add_argument("--max-batches", type=int, help="Prediction smoke test only; marked partial_smoke")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if not re.fullmatch(r"cpu|cuda:\d+", args.device):
        parser.error("--device must be cpu or cuda:N")
    if args.checkpoint and args.action not in ("predict", "evaluate"):
        parser.error("--checkpoint is only supported by predict/evaluate")
    if args.action in ("predict", "evaluate") and not args.checkpoint:
        parser.error("predict requires --checkpoint pointing to your trained weight or training output directory; run train first")
    if args.action == "list":
        runs = [run for run in configured_runs() if (not args.dataset or run["dataset"] == args.dataset)
                and (not args.city or run["city"] == args.city)
                and (args.horizon is None or run["horizon"] == args.horizon)
                and (args.graph_k is None or (run["dataset"] == "chinaair" and run["graph_k"] == args.graph_k))]
        for run in runs:
            selectors = ["--dataset", run["dataset"], "--horizon", str(run["horizon"])]
            if args.graph_k is not None:
                selectors += ["--graph-k", str(run["graph_k"])]
            if run["city"]:
                selectors += ["--city", run["city"]]
            slug = "_".join(str(run[k]) for k in ("dataset", "city", "horizon") if run[k] is not None)
            train_dir = f"outputs/reproduction/train/{slug}"
            for action in ("train", "predict"):
                command = ["python", "run.py", action, *selectors, "--device", args.device,
                           "--output-dir", train_dir if action == "train" else f"outputs/reproduction/predict/{slug}"]
                if action == "predict":
                    command += ["--checkpoint", train_dir]
                print(shlex.join(command))
        print(f"# {len(runs)} experiments; two commands per experiment")
        return
    if not args.dataset or args.horizon is None or not args.output_dir:
        parser.error("train/predict require --dataset, --horizon and --output-dir")
    if args.action == "train" and (args.max_batches is not None or args.precision != "native" or args.split != "test"):
        parser.error("--max-batches, --precision and --split are prediction options")
    run = select_run(args.dataset, args.city, args.horizon, args.graph_k)
    out = output_path(args.output_dir, args.checkpoint)
    if args.action == "train":
        cwd, command = train_command(run, args.device, out)
    else:
        cwd = ROOT
        command = [sys.executable, "-B", "-m", "engine.evaluate",
            "--dataset", args.dataset, "--horizon", str(args.horizon), "--device", args.device,
            "--output-dir", str(out), "--split", args.split, "--precision", args.precision]
        if args.checkpoint:
            command += ["--checkpoint", str(checkpoint_path(args.checkpoint))]
        if args.graph_k is not None:
            command += ["--graph-k", str(args.graph_k)]
        if args.city:
            command += ["--city", args.city]
        if args.max_batches is not None:
            command += ["--max-batches", str(args.max_batches)]
    plan = dict(action=args.action, model="CARAMOE", **run, cwd=str(cwd), command=command,
                native_command=shlex.join(command), output_dir=str(out), dry_run=args.dry_run)
    if args.checkpoint:
        plan["checkpoint_source"] = "user_trained"
        plan["checkpoint"] = str(checkpoint_path(args.checkpoint))
        plan["checkpoint_sha256"] = None
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    require_data(run)
    import torch
    if args.device.startswith("cuda:") and (not torch.cuda.is_available() or int(args.device.split(":")[1]) >= torch.cuda.device_count()):
        raise RuntimeError(f"Requested device is unavailable: {args.device}")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(command, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    main()
