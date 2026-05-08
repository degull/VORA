from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImagePair:
    degraded: Path
    clean: Path


def _list_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def _pair_same_name(degraded_dir: Path, clean_dir: Path) -> list[ImagePair]:
    clean_by_name = {path.name: path for path in _list_images(clean_dir)}
    pairs = []
    for degraded in _list_images(degraded_dir):
        clean = clean_by_name.get(degraded.name)
        if clean is not None:
            pairs.append(ImagePair(degraded=degraded, clean=clean))
    return pairs


def _pair_gopro(root: Path, split: str) -> list[ImagePair]:
    split_root = root / split
    pairs = []
    for blur_dir in sorted(split_root.glob("*/blur")):
        sharp_dir = blur_dir.parent / "sharp"
        if sharp_dir.exists():
            pairs.extend(_pair_same_name(blur_dir, sharp_dir))

    if pairs:
        return pairs

    csv_path = root / f"gopro_{split}_pairs.csv"
    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            degraded = Path(row["dist_img"])
            clean = Path(row["ref_img"])
            if degraded.exists() and clean.exists():
                pairs.append(ImagePair(degraded=degraded, clean=clean))
    return pairs


def _resolve_csv_path(path_text: str, dataset_root: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path

    normalized = path_text.replace("\\", "/")
    if "/Data/" in normalized:
        tail = normalized.split("/Data/", 1)[1]
        candidate = dataset_root / "Data" / tail
        if candidate.exists():
            return candidate
    return path


def _pair_from_csv(root: Path, csv_name: str) -> list[ImagePair]:
    csv_path = root / csv_name
    pairs = []
    if not csv_path.exists():
        return pairs
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            degraded = _resolve_csv_path(row["dist_img"], root)
            clean = _resolve_csv_path(row["ref_img"], root)
            if degraded.exists() and clean.exists():
                pairs.append(ImagePair(degraded=degraded, clean=clean))
    return pairs


def collect_pairs(dataset: str, data_root: Path, split: str) -> list[ImagePair]:
    dataset = dataset.lower()

    if dataset == "rain100h":
        split_root = data_root / "rain100H" / split
        return _pair_same_name(split_root / "rain", split_root / "norain")

    if dataset == "csd":
        split_name = "Train" if split == "train" else "Test"
        split_root = data_root / "CSD" / split_name
        return _pair_same_name(split_root / "Snow", split_root / "Gt")

    if dataset == "gopro":
        return _pair_gopro(data_root / "GOPRO_Large", split)

    if dataset == "reside6k":
        split_root = data_root / "RESIDE-6K" / split
        return _pair_same_name(split_root / "hazy", split_root / "GT")

    if dataset == "sidd":
        csv_name = "sidd_pairs.csv" if split == "train" else "sidd_test_pairs.csv"
        return _pair_from_csv(data_root / "SIDD", csv_name)

    raise ValueError(f"Unsupported dataset: {dataset}")


class PairedImageDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        data_root: str | Path,
        split: str = "train",
        crop_size: int = 128,
        max_samples: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.data_root = Path(data_root)
        self.split = split
        self.crop_size = crop_size
        self.pairs = collect_pairs(dataset, self.data_root, split)
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]
        if not self.pairs:
            raise RuntimeError(f"No image pairs found for {dataset} {split} under {self.data_root}.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        degraded = Image.open(pair.degraded).convert("RGB")
        clean = Image.open(pair.clean).convert("RGB")

        degraded_tensor = F.to_tensor(degraded)
        clean_tensor = F.to_tensor(clean)
        degraded_tensor, clean_tensor = self._aligned_crop(degraded_tensor, clean_tensor)

        return {
            "input": degraded_tensor,
            "target": clean_tensor,
            "name": pair.degraded.name,
        }

    def _aligned_crop(self, degraded: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, height, width = degraded.shape
        crop = min(self.crop_size, height, width)
        if self.split != "train":
            return degraded[:, :crop, :crop], clean[:, :crop, :crop]

        top = random.randint(0, height - crop) if height > crop else 0
        left = random.randint(0, width - crop) if width > crop else 0
        return (
            degraded[:, top : top + crop, left : left + crop],
            clean[:, top : top + crop, left : left + crop],
        )
