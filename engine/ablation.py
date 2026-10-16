from __future__ import annotations


import time
from typing import Dict

from torch.utils.data import DataLoader

from engine.residual import CARAMOETrainer


class CARAMOEAblationTrainer(CARAMOETrainer):

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        started = time.perf_counter()
        result = super().train_epoch(loader, epoch)
        result["wall_seconds"] = time.perf_counter() - started
        return result

    def evaluate(self, loader: DataLoader, mode: str) -> Dict[str, float]:
        started = time.perf_counter()
        result = super().evaluate(loader, mode)
        result["wall_seconds"] = time.perf_counter() - started
        return result
