import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main comparison table experiments.")
    parser.add_argument("--datasets", nargs="+", default=["rain100h", "csd", "gopro", "reside6k"])
    parser.add_argument("--methods", nargs="+", default=["full_ft", "frozen", "lora", "vora_v1"])
    parser.add_argument("--backbone", default="swinir_official")
    parser.add_argument("--swinir-size", default="tiny")
    parser.add_argument("--target", default="all")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-samples", type=int, default=2)
    parser.add_argument("--max-val-samples", type=int, default=1)
    parser.add_argument("--results-csv", default="outputs/logs/table1_main_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.results_csv)
    if csv_path.exists():
        csv_path.unlink()

    for dataset in args.datasets:
        for method in args.methods:
            command = [
                sys.executable,
                "train.py",
                "--dataset",
                dataset,
                "--method",
                method,
                "--backbone",
                args.backbone,
                "--swinir-size",
                args.swinir_size,
                "--target",
                args.target,
                "--rank",
                str(args.rank),
                "--steps",
                str(args.steps),
                "--crop-size",
                str(args.crop_size),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--max-train-samples",
                str(args.max_train_samples),
                "--max-val-samples",
                str(args.max_val_samples),
                "--results-csv",
                args.results_csv,
            ]
            print("")
            print(f"Running {dataset} / {method}")
            subprocess.run(command, check=True)

    print("")
    print(f"Saved table rows to {csv_path}")


if __name__ == "__main__":
    main()
