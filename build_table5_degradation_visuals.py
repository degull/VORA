import argparse
from pathlib import Path

import torch
from torchvision.transforms import functional as F
from torchvision.utils import make_grid, save_image

from datasets.composite_degradation_dataset import CompositeDegradationDataset
from datasets.paired_image_dataset import PairedImageDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export visual examples for degradation interaction experiments.")
    parser.add_argument("--dataset", default="gopro")
    parser.add_argument("--data-root", default=r"E:\restormer+volterra\data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--added-degradation", default="noise")
    parser.add_argument("--intensities", nargs="+", type=float, default=[0.02, 0.04, 0.08])
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/figures/table5_degradation_interaction")
    return parser.parse_args()


def label_strip(width: int, height: int = 10) -> torch.Tensor:
    return torch.ones(3, height, width)


def residual_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(a - b)
    max_value = torch.clamp(diff.max(), min=1e-6)
    return torch.clamp(diff / max_value, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = PairedImageDataset(
        dataset=args.dataset,
        data_root=args.data_root,
        split=args.split,
        crop_size=args.crop_size,
        max_samples=args.num_samples,
    )

    for sample_idx in range(min(args.num_samples, len(base))):
        base_sample = base[sample_idx]
        original = base_sample["input"]
        target = base_sample["target"]
        rows = []
        for intensity in args.intensities:
            composite_dataset = CompositeDegradationDataset(
                dataset=args.dataset,
                data_root=args.data_root,
                split=args.split,
                crop_size=args.crop_size,
                max_samples=args.num_samples,
                added_degradation=args.added_degradation,
                degradation_intensity=intensity,
            )
            composite = composite_dataset[sample_idx]["input"]
            residual = residual_map(composite, original)
            rows.extend([original, composite, residual, target])

        grid = make_grid(rows, nrow=4, padding=4, pad_value=1.0)
        save_image(grid, output_dir / f"{args.dataset}_{args.added_degradation}_sample{sample_idx + 1}.png")

        for intensity in args.intensities:
            composite_dataset = CompositeDegradationDataset(
                dataset=args.dataset,
                data_root=args.data_root,
                split=args.split,
                crop_size=args.crop_size,
                max_samples=args.num_samples,
                added_degradation=args.added_degradation,
                degradation_intensity=intensity,
            )
            composite = composite_dataset[sample_idx]["input"]
            intensity_text = str(intensity).replace(".", "p")
            F.to_pil_image(composite).save(output_dir / f"{args.dataset}_{args.added_degradation}_i{intensity_text}_sample{sample_idx + 1}.png")

    print(f"Saved visual examples to {output_dir}")


if __name__ == "__main__":
    main()
