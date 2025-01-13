from __future__ import annotations

import torch
from torch.utils.data import Dataset

from train_jalb.quantization import pixel_art_transform


class PixelArtDataset(Dataset):
    def __init__(self, hf_dataset, prompt: str, size: int = 512) -> None:
        self.ds = hf_dataset
        self.default_prompt = prompt
        self.transform = pixel_art_transform(size)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        example = self.ds[idx]
        prompt = example.get("text") or self.default_prompt
        return {
            "image": self.transform(example["image"].convert("RGB")),
            "prompt": prompt,
        }


def collate(examples: list[dict]) -> dict:
    return {
        "images": torch.stack([example["image"] for example in examples]),
        "prompts": [example["prompt"] for example in examples],
    }
