from __future__ import annotations

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

GRAY_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
LUMA_WEIGHTS = (0.299, 0.587, 0.114)


class SixGrayQuantizer(nn.Module):
    """Snap pixels in [0, 1] to six fixed gray levels."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("levels", torch.tensor(GRAY_LEVELS))
        self.register_buffer("luma", torch.tensor(LUMA_WEIGHTS).view(3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-3] == 3:
            gray = (x * self.luma.to(dtype=x.dtype, device=x.device)).sum(dim=-3)
        else:
            gray = x.squeeze(-3)

        levels = self.levels.to(dtype=x.dtype, device=x.device)
        idx = (gray.unsqueeze(-1) - levels).abs().argmin(dim=-1)
        snapped = levels[idx].unsqueeze(-3)
        return snapped.expand(*snapped.shape[:-3], 3, *snapped.shape[-2:])


def pixel_art_transform(size: int = 512) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((size // 8, size // 8), interpolation=InterpolationMode.NEAREST),
            transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            SixGrayQuantizer(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
