import argparse
import csv
import subprocess
import sys
from pathlib import Path


RUNS = [
    ("lora", 8),
    ("vora_v1", 4),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backbone-wise parameter-matched adapter comparisons.")
    parser.add_argument("--backbones", nargs="+", default=["uformer", "hat", "adair", "dfpir_restormer", "mambairv2"])
    parser.add_argument("--dataset", default="rain100h")
    parser.add_argument("--swinir-size", default="base")
    parser.add_argument("--backbone-size", default="base")
    parser.add_argument("--target", default="all")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--results-csv", default="outputs/logs/table3_backbone_param_matched.csv")
    parser.add_argument("--checkpoint-dir", default="checkpoints/table3_backbone_param_matched")
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-dir", default="outputs/logs/table3_backbone_param_matched_runs")
    parser.add_argument("--reset-results", action="store_true")
    parser.add_argument("--no-skip-completed", action="store_true")
    return parser.parse_args()


def completed_runs(csv_path: Path) -> set[tuple[str, str, str, str]]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", newline="") as handle:
        return {
            (row["backbone"], row["dataset"], row["method"], row["rank"])
            for row in csv.DictReader(handle)
        }


def run_and_log(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.results_csv)
    if args.reset_results and csv_path.exists():
        csv_path.unlink()

    completed = completed_runs(csv_path)
    for backbone in args.backbones:
        for method, rank in RUNS:
            key = (backbone, args.dataset, method, str(rank))
            if not args.no_skip_completed and key in completed:
                print(f"Skipping completed Table 3 {backbone} / {args.dataset} / {method} / r{rank}")
                continue

            command = [
                sys.executable,
                "train.py",
                "--dataset",
                args.dataset,
                "--method",
                method,
                "--backbone",
                backbone,
                "--swinir-size",
                args.swinir_size,
                "--backbone-size",
                args.backbone_size,
                "--target",
                args.target,
                "--rank",
                str(rank),
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
                "--save-interval",
                str(args.save_interval),
                "--auto-resume",
            ]
            print("")
            print(f"Running Table 3 {backbone} / {args.dataset} / {method} / r{rank}")
            log_path = Path(args.log_dir) / f"{backbone}_{args.dataset}_{method}_r{rank}.log"
            run_and_log(command, log_path)

            completed = completed_runs(csv_path)
            if key not in completed:
                raise RuntimeError(
                    f"Run finished but no result row was written for {key}. "
                    f"Check log: {log_path}"
                )

    print("")
    print(f"Saved Table 3 rows to {csv_path}")


if __name__ == "__main__":
    main()
