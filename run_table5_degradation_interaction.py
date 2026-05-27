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
    parser = argparse.ArgumentParser(description="Run degradation interaction experiments.")
    parser.add_argument("--dataset", default="gopro")
    parser.add_argument("--added-degradation", default="noise")
    parser.add_argument("--intensities", nargs="+", type=float, default=[0.02, 0.04, 0.06, 0.08])
    parser.add_argument("--backbone", default="swinir_official")
    parser.add_argument("--swinir-size", default="base")
    parser.add_argument("--target", default="all")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--results-csv", default="outputs/logs/table5_degradation_interaction.csv")
    parser.add_argument("--checkpoint-dir", default="checkpoints/table5_degradation_interaction")
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-dir", default="outputs/logs/table5_degradation_interaction_runs")
    parser.add_argument("--reset-results", action="store_true")
    parser.add_argument("--no-skip-completed", action="store_true")
    return parser.parse_args()


def intensity_key(value: str) -> str:
    return f"{float(value):.6f}"


def completed_runs(csv_path: Path) -> set[tuple[str, str, str, str, str]]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", newline="") as handle:
        return {
            (
                row["dataset"],
                row.get("added_degradation", "none"),
                intensity_key(row.get("degradation_intensity", "-1")),
                row["method"],
                row["rank"],
            )
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
    for intensity in args.intensities:
        for method, rank in RUNS:
            key = (
                args.dataset,
                args.added_degradation,
                intensity_key(str(intensity)),
                method,
                str(rank),
            )
            if not args.no_skip_completed and key in completed:
                print(f"Skipping completed Table 5 {args.dataset} + {args.added_degradation}={intensity} / {method} / r{rank}")
                continue

            command = [
                sys.executable,
                "train.py",
                "--dataset",
                args.dataset,
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
                args.added_degradation,
                "--degradation-intensity",
                str(intensity),
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
            intensity_text = str(intensity).replace(".", "p")
            print("")
            print(f"Running Table 5 {args.dataset} + {args.added_degradation}={intensity} / {method} / r{rank}")
            log_path = Path(args.log_dir) / f"{args.dataset}_{args.added_degradation}_i{intensity_text}_{method}_r{rank}.log"
            run_and_log(command, log_path)

            completed = completed_runs(csv_path)
            if key not in completed:
                raise RuntimeError(
                    f"Run finished but no result row was written for {key}. "
                    f"Check log: {log_path}"
                )

    print("")
    print(f"Saved Table 5 rows to {csv_path}")


if __name__ == "__main__":
    main()
