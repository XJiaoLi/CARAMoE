from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AugmentResult:
    values: Tensor
    mask: Tensor
    additionally_hidden_channels: Tensor


class ObservationMaskAugmenter:

    def __init__(
        self,
        trailing_block_prob: float = 0.25,
        trailing_block_min: int = 3,
        trailing_block_max: int = 24,
        full_channel_drop_prob: float = 0.05,
        station_drop_prob: float = 0.02,
        random_point_drop_prob: float = 0.03,
    ) -> None:
        self.trailing_block_prob = trailing_block_prob
        self.trailing_block_min = trailing_block_min
        self.trailing_block_max = trailing_block_max
        self.full_channel_drop_prob = full_channel_drop_prob
        self.station_drop_prob = station_drop_prob
        self.random_point_drop_prob = random_point_drop_prob

    def __call__(self, x: Tensor, mask: Tensor) -> AugmentResult:
        mask = mask.bool()
        b, t, n, p = mask.shape
        augmented = mask.clone()
        hidden_channels = torch.zeros((b, n, p), dtype=torch.bool, device=mask.device)

        if self.trailing_block_prob > 0:
            selected = torch.rand((b, n, p), device=mask.device) < self.trailing_block_prob
            max_len = min(self.trailing_block_max, t)
            min_len = min(self.trailing_block_min, max_len)
            length = int(torch.randint(min_len, max_len + 1, (1,), device=mask.device).item())
            augmented[:, -length:] &= ~selected[:, None]
            hidden_channels |= selected

        if self.full_channel_drop_prob > 0:
            selected = torch.rand((b, n, p), device=mask.device) < self.full_channel_drop_prob
            augmented &= ~selected[:, None]
            hidden_channels |= selected

        if self.station_drop_prob > 0:
            selected_station = torch.rand((b, n), device=mask.device) < self.station_drop_prob
            augmented &= ~selected_station[:, None, :, None]
            hidden_channels |= selected_station[:, :, None]

        if self.random_point_drop_prob > 0:
            point_drop = torch.rand(mask.shape, device=mask.device) < self.random_point_drop_prob
            augmented &= ~point_drop

        values = x.masked_fill(~augmented, 0.0)
        return AugmentResult(values, augmented, hidden_channels)
