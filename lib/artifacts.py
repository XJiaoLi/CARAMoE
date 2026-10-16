from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class PredictionArtifactWriter:

    def __init__(self) -> None:
        self.predictions: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.target_starts: list[np.ndarray] = []
        self.target_hours: list[np.ndarray] = []
        self.batch_offsets = [0]

    def append(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        target_start: np.ndarray,
        target_hours: np.ndarray,
    ) -> None:
        prediction = np.asarray(prediction, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.bool_)
        if prediction.shape != target.shape or prediction.shape != mask.shape:
            raise ValueError("Prediction, target, and mask shapes must match.")
        self.predictions.append(prediction)
        self.targets.append(target)
        self.masks.append(mask)
        self.target_starts.append(np.asarray(target_start, dtype=np.int64))
        self.target_hours.append(np.asarray(target_hours, dtype=np.int64))
        self.batch_offsets.append(self.batch_offsets[-1] + prediction.shape[0])

    def save(self, path: str | Path, metadata: dict[str, object]) -> Path:
        if not self.predictions:
            raise RuntimeError("No prediction batches were collected.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "wb") as handle:
            np.savez_compressed(
                handle,
                prediction=np.concatenate(self.predictions, axis=0),
                target=np.concatenate(self.targets, axis=0),
                mask=np.concatenate(self.masks, axis=0),
                target_start_index=np.concatenate(self.target_starts, axis=0),
                target_unix_hours=np.concatenate(self.target_hours, axis=0),
                batch_offsets=np.asarray(self.batch_offsets, dtype=np.int64),
                metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )
        temporary.replace(path)
        return path
