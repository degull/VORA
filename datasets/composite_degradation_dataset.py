from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torchvision.transforms import functional as F

from .paired_image_dataset import PairedImageDataset


COMPOSITE_CHOICES = ("none", "haze", "blur", "noise")


def _seed_from_name(name: str, split: str, degradation: str) -> int:
    digest = hashlib.sha256(f"{split}:{degradation}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def add_haze(x: torch.Tensor, transmission: float = 0.72, airlight: float = 1.0) -> torch.Tensor:
    return torch.clamp(x * transmission + airlight * (1.0 - transmission), 0.0, 1.0)


def add_blur(x: torch.Tensor, kernel_size: int = 7, sigma: float = 1.4) -> torch.Tensor:
    return F.gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def add_noise(x: torch.Tensor, name: str, split: str, std: float = 0.04) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed_from_name(name, split, "noise"))
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype) * std
    return torch.clamp(x + noise, 0.0, 1.0)


def apply_composite_degradation(
    x: torch.Tensor,
    degradation: str,
    name: str,
    split: str,
) -> torch.Tensor:
    if degradation == "none":
        return x
    if degradation == "haze":
        return add_haze(x)
    if degradation == "blur":
        return add_blur(x)
    if degradation == "noise":
        return add_noise(x, name=name, split=split)
    raise ValueError(f"Unsupported composite degradation: {degradation}")


class CompositeDegradationDataset(PairedImageDataset):
    def __init__(
        self,
        dataset: str,
        data_root: str | Path,
        split: str = "train",
        crop_size: int = 128,
        max_samples: int | None = None,
        added_degradation: str = "none",
    ) -> None:
        if added_degradation not in COMPOSITE_CHOICES:
            raise ValueError(f"Unsupported added degradation: {added_degradation}")
        super().__init__(
            dataset=dataset,
            data_root=data_root,
            split=split,
            crop_size=crop_size,
            max_samples=max_samples,
        )
        self.added_degradation = added_degradation

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = super().__getitem__(index)
        name = str(sample["name"])
        sample["input"] = apply_composite_degradation(
            sample["input"],
            degradation=self.added_degradation,
            name=name,
            split=self.split,
        )
        return sample
