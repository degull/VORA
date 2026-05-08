import argparse
import subprocess
import sys
from pathlib import Path


TASK_DATASETS = {
    "deraining": "rain100h",
    "desnowing": "csd",
    "denoising": "sidd",
    "deblurring": "gopro",
    "dehazing": "reside6k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Table 2 task-generalization experiments.")
    parser.add_argument("--methods", nargs="+", default=["lora", "vora_v1", "vora_token", "vora_full"])
    parser.add_argument("--backbone", default="swinir_official")
    parser.add_argument("--swinir-size", default="tiny")
    parser.add_argument("--target", default="all")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-samples", type=int, default=1)
    parser.add_argument("--max-val-samples", type=int, default=1)
    parser.add_argument("--results-csv", default="outputs/logs/table2_task_generalization.csv")
    parser.add_argument("--checkpoint-dir", default="checkpoints/table2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.results_csv)
    if csv_path.exists():
        csv_path.unlink()

    for task, dataset in TASK_DATASETS.items():
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
                "--checkpoint-dir",
                args.checkpoint_dir,
            ]
            print("")
            print(f"Running Table 2 {task} ({dataset}) / {method}")
            subprocess.run(command, check=True)

    print("")
    print(f"Saved Table 2 rows to {csv_path}")


if __name__ == "__main__":
    main()
