from __future__ import annotations

import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lib.config import ExperimentConfig, checkpoint_config_fingerprint

GROUP_KEYS = ("method_label", "input_profile", "input_window", "output_window")


def load_seed_results(output_root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(output_root).rglob("final_metrics.json")):
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["result_path"] = str(path)
        rows.append(row)
    return rows

def _numeric_test_metrics(rows: Iterable[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key.startswith("test_") and isinstance(value, (int, float)) and math.isfinite(float(value)):
                keys.add(key)
    return sorted(keys)

def summarize(output_root: str | Path, minimum_seeds: int = 5) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = load_seed_results(output_root)
    if not rows:
        result = {
            "output_root": str(output_root),
            "minimum_seeds": minimum_seeds,
            "all_groups_complete": False,
            "groups": [],
        }
        with open(output_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    seed_columns = sorted({key for row in rows for key in row})
    with open(output_root / "seed_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=seed_columns)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in GROUP_KEYS)].append(row)

    summary_rows: list[dict[str, Any]] = []
    group_results: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        metadata = dict(zip(GROUP_KEYS, group_key))
        seed_values = [int(row["seed"]) for row in group_rows]
        seeds = sorted(set(seed_values))
        if len(seeds) != len(seed_values):
            raise ValueError(
                "Duplicate final_metrics.json files were found for "
                f"{metadata}; each seed must contribute exactly one result."
            )
        group_record: dict[str, Any] = {
            **metadata,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "complete": len(seeds) >= minimum_seeds,
            "metrics": {},
        }
        for metric in _numeric_test_metrics(group_rows):
            values = np.asarray(
                [float(row[metric]) for row in group_rows if metric in row and math.isfinite(float(row[metric]))],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            group_record["metrics"][metric] = {"mean": mean, "std": std, "n": int(values.size)}
            summary_rows.append(
                {
                    **metadata,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": mean,
                    "std": std,
                    "mean_plus_minus_std": f"{mean:.6f} ± {std:.6f}",
                    "complete": len(seeds) >= minimum_seeds,
                }
            )
        group_results.append(group_record)

    with open(output_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[*GROUP_KEYS, "metric", "n", "mean", "std", "mean_plus_minus_std", "complete"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    result = {
        "output_root": str(output_root),
        "minimum_seeds": minimum_seeds,
        "all_groups_complete": bool(group_results) and all(group["complete"] for group in group_results),
        "groups": group_results,
    }
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result

def validate_completed_result(
    path: Path,
    config: ExperimentConfig,
    variant: str,
    profile: str,
    seed: int,
) -> None:
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)
    expected = copy.deepcopy(config)
    expected.train.training_variant = variant
    expected.model.input_profile = profile
    expected.train.seed = seed
    expected.train.output_dir = str(path.parent)
    expected.validate()
    expected_values = {
        "method_label": expected.method_label,
        "training_variant": variant,
        "input_profile": profile,
        "seed": seed,
        "input_window": expected.data.input_window,
        "output_window": expected.data.output_window,
        "train_step": expected.data.train_step,
        "eval_step": expected.data.eval_step,
        "config_fingerprint": checkpoint_config_fingerprint(expected),
    }
    differences = [
        f"{key}: expected={value!r}, found={result.get(key)!r}"
        for key, value in expected_values.items()
        if result.get(key) != value
    ]
    if differences:
        raise ValueError(
            f"Completed result is incompatible with its benchmark slot: {path}: "
            + "; ".join(differences)
        )

