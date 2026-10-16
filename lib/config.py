from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


@dataclass
class DataConfig:
    data_dir: str = "./data"
    metadata_path: str = "./data/selected_nodes_metadata.csv"
    scaler_path: str = "./data/scaler.csv"
    graph_cache_path: str = "./data/caramoe_graph_k10.npz"
    capability_cache_path: str = "./data/capability_train_2021_2023.npz"
    train_years: tuple[int, ...] = (2021, 2022, 2023)
    val_years: tuple[int, ...] = (2024,)
    test_years: tuple[int, ...] = (2025,)
    input_window: int = 24
    output_window: int = 12
    train_step: int = 12
    eval_step: int = 12
    cross_year_windows: bool = True
    num_workers: int = 0
    max_stations: int | None = None
    station_subset_path: str | None = None


@dataclass
class ModelConfig:
    hidden_dim: int = 48
    station_dim: int = 24
    pollutant_dim: int = 16
    time_dim: int = 16
    router_dim: int = 32
    num_pollutants: int = 6
    knn_k: int = 10
    spatial_chunk_size: int = 256
    dropout: float = 0.1
    station_id_dropout: float = 0.3
    use_station_id: bool = True
    decoder_layers: int = 2
    input_profile: str = "full"


@dataclass
class AugmentConfig:
    enabled: bool = True
    trailing_block_prob: float = 0.25
    trailing_block_min: int = 3
    trailing_block_max: int = 24
    full_channel_drop_prob: float = 0.05
    station_drop_prob: float = 0.02
    random_point_drop_prob: float = 0.03


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "auto"
    epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    amp: bool = True
    log_every: int = 20
    scheduler_patience: int = 5
    early_stopping_patience: int | None = None
    output_dir: str = "./outputs/caramoe"
    training_variant: str = "C"
    expert_aux_weight: float = 0.15
    safe_weight: float = 0.10
    route_consistency_weight: float = 0.05
    high_reliability_threshold: float = 0.75
    safe_margin_normalized: float = 0.0
    run_consistency_every: int = 4
    max_train_batches: int | None = None
    max_eval_batches: int | None = None


@dataclass
class BenchmarkConfig:

    seeds: tuple[int, ...] = (42, 123, 456, 789, 1024)
    training_variants: tuple[str, ...] = ("A", "C")
    input_profiles: tuple[str, ...] = ("strict", "full")
    output_root: str = "./outputs/benchmark"


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    augment: AugmentConfig
    train: TrainConfig
    benchmark: BenchmarkConfig

    @classmethod
    def default(cls) -> "ExperimentConfig":
        return cls(DataConfig(), ModelConfig(), AugmentConfig(), TrainConfig(), BenchmarkConfig())

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f) if path.suffix.lower() == ".json" else yaml.safe_load(f)
            raw = raw or {}
        base = cls.default()
        return cls(
            data=DataConfig(**{**asdict(base.data), **raw.get("data", {})}),
            model=ModelConfig(**{**asdict(base.model), **raw.get("model", {})}),
            augment=AugmentConfig(**{**asdict(base.augment), **raw.get("augment", {})}),
            train=TrainConfig(**{**asdict(base.train), **raw.get("train", {})}),
            benchmark=BenchmarkConfig(**{**asdict(base.benchmark), **raw.get("benchmark", {})}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def method_label(self) -> str:
        return "CARAMOE" if self.train.training_variant.upper() == "C" else f"CARAMOE-{self.train.training_variant.upper()}"

    def validate(self) -> None:
        variant = self.train.training_variant.upper()
        if variant not in {"A", "C"}:
            raise ValueError(f"training_variant must be A or C, got {self.train.training_variant!r}.")
        self.train.training_variant = variant

        profile = self.model.input_profile.lower()
        if profile not in {"strict", "full"}:
            raise ValueError(f"input_profile must be strict or full, got {self.model.input_profile!r}.")
        self.model.input_profile = profile

        if self.data.output_window not in {3, 6, 12, 18, 24}:
            raise ValueError("output_window must be one of 3, 6, 12, 18 or 24 hours.")
        if self.train.epochs <= 0 or self.train.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.train.scheduler_patience < 0:
            raise ValueError("scheduler_patience must be non-negative.")
        if self.train.early_stopping_patience is not None and self.train.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive or null.")

        variants = tuple(str(v).upper() for v in self.benchmark.training_variants)
        profiles = tuple(str(v).lower() for v in self.benchmark.input_profiles)
        if not variants or any(v not in {"A", "C"} for v in variants):
            raise ValueError("benchmark.training_variants must contain only A and/or C.")
        if not profiles or any(v not in {"strict", "full"} for v in profiles):
            raise ValueError("benchmark.input_profiles must contain only strict and/or full.")
        if not self.benchmark.seeds:
            raise ValueError("benchmark.seeds cannot be empty.")
        self.benchmark.training_variants = variants
        self.benchmark.input_profiles = profiles
        self.benchmark.seeds = tuple(int(seed) for seed in self.benchmark.seeds)


def checkpoint_config_payload(config: ExperimentConfig | Mapping[str, Any]) -> Dict[str, Any]:

    raw = config.to_dict() if isinstance(config, ExperimentConfig) else config
    payload = json.loads(json.dumps(raw))
    payload.pop("benchmark", None)
    train = payload.get("train", {})
    train.pop("output_dir", None)
    return payload


def checkpoint_config_fingerprint(config: ExperimentConfig | Mapping[str, Any]) -> str:
    serialized = json.dumps(
        checkpoint_config_payload(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_checkpoint_config(
    current: ExperimentConfig,
    saved_config: Mapping[str, Any],
    saved_fingerprint: str | None = None,
) -> None:
    current_payload = checkpoint_config_payload(current)
    saved_payload = checkpoint_config_payload(saved_config)
    current_fingerprint = checkpoint_config_fingerprint(current_payload)
    expected_fingerprint = saved_fingerprint or checkpoint_config_fingerprint(saved_payload)
    if current_fingerprint == expected_fingerprint and current_payload == saved_payload:
        return

    differences: list[str] = []

    def visit(prefix: str, left: Any, right: Any) -> None:
        if len(differences) >= 12:
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                visit(f"{prefix}.{key}" if prefix else str(key), left.get(key), right.get(key))
        elif left != right:
            differences.append(f"{prefix}: current={left!r}, checkpoint={right!r}")

    visit("", current_payload, saved_payload)
    detail = "; ".join(differences) if differences else "configuration fingerprint mismatch"
    raise ValueError(f"Checkpoint configuration is incompatible: {detail}")
