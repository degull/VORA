import argparse
import csv
import subprocess
import sys
from pathlib import Path


COMPOSITES = [
    ("rain100h", "haze"),
    ("rain100h", "blur"),
    ("gopro", "noise"),
    ("reside6k", "noise"),
]
RUNS = [
    ("lora", 8),
    ("vora_v1", 4),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run composite degradation robustness experiments.")
    parser.add_argument("--backbone", default="swinir_official")
    parser.add_argument("--swinir-size", default="base")
    parser.add_argument("--target", default="all")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--results-csv", default="outputs/logs/table4_composite_degradation.csv")
    parser.add_argument("--checkpoint-dir", default="checkpoints/table4_composite_degradation")
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-dir", default="outputs/logs/table4_composite_degradation_runs")
    parser.add_argument("--reset-results", action="store_true")
    parser.add_argument("--no-skip-completed", action="store_true")
    return parser.parse_args()


def completed_runs(csv_path: Path) -> set[tuple[str, str, str, str]]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", newline="") as handle:
        return {
            (row["dataset"], row.get("added_degradation", "none"), row["method"], row["rank"])
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
    for dataset, added_degradation in COMPOSITES:
        for method, rank in RUNS:
            key = (dataset, added_degradation, method, str(rank))
            if not args.no_skip_completed and key in completed:
                print(f"Skipping completed Table 4 {dataset} + {added_degradation} / {method} / r{rank}")
                continue

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
                str(rank),
                "--added-degradation",
                added_degradation,
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
            print(f"Running Table 4 {dataset} + {added_degradation} / {method} / r{rank}")
            log_path = Path(args.log_dir) / f"{dataset}_{added_degradation}_{method}_r{rank}.log"
            run_and_log(command, log_path)

            completed = completed_runs(csv_path)
            if key not in completed:
                raise RuntimeError(
                    f"Run finished but no result row was written for {key}. "
                    f"Check log: {log_path}"
                )

    print("")
    print(f"Saved Table 4 rows to {csv_path}")


if __name__ == "__main__":
    main()
