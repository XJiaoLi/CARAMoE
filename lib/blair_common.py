from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .artifacts import PredictionArtifactWriter
from lib.paths import METADATA
from data_provider.blair import KDDWindowDataset
from preprocess.blair import POLLUTANTS


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_scaler(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if tuple(frame["pollutant"]) != POLLUTANTS:
        raise ValueError(f"Scaler pollutant order must be {POLLUTANTS}.")
    return (
        frame["mean"].to_numpy(dtype=np.float32),
        frame["std"].to_numpy(dtype=np.float32),
    )


def load_manifest(path: str | Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def city_paths(processed_root: str | Path, city: str) -> dict[str, Path]:
    city_key = city.casefold()
    if city_key not in {"beijing", "london"}:
        raise ValueError("city must be Beijing or London")
    data_root = Path(processed_root).resolve()
    root = data_root / city_key
    h5 = root / f"{city_key}.h5"
    if not h5.exists():
        h5 = data_root / f"{city_key}.h5"
        root = METADATA / "BLAir" / city_key
    return {
        "root": root,
        "h5": h5,
        "manifest": root / "split_manifest.json",
        "metadata": root / "metadata.csv",
        "scaler": root / "scaler.csv",
        "capability": root / "capability_train.npz",
    }


def build_datasets(
    paths: dict[str, Path],
    mean: np.ndarray,
    std: np.ndarray,
    horizon: int,
) -> dict[str, KDDWindowDataset]:
    common = dict(
        h5_path=paths["h5"],
        manifest_path=paths["manifest"],
        mean=mean,
        std=std,
        input_window=24,
        output_window=horizon,
        step=horizon,
    )
    return {
        split: KDDWindowDataset(split=split, **common) for split in ("train", "val", "test")
    }


def make_loader(
    dataset: KDDWindowDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    pin_memory = num_workers > 0 and torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


@torch.no_grad()
def export_predictions(
    loader: DataLoader,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    predict: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    output_path: str | Path,
    metadata: dict[str, object],
    max_batches: int | None = None,
) -> Path:
    writer = PredictionArtifactWriter()
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, 1, 1, 6)
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, 1, 1, 6)
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        device_batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        prediction_norm = predict(device_batch)
        target_norm = device_batch["y"]
        prediction_real = prediction_norm.float() * (std_tensor + 1e-5) + mean_tensor
        target_real = target_norm.float() * (std_tensor + 1e-5) + mean_tensor
        writer.append(
            prediction_real.detach().cpu().numpy(),
            target_real.detach().cpu().numpy(),
            raw_batch["y_mask"].cpu().numpy(),
            raw_batch["target_start_index"].cpu().numpy(),
            raw_batch["target_unix_hours"].cpu().numpy(),
        )
    return writer.save(output_path, metadata)
