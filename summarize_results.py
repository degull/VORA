import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a paper-style Markdown table from result CSV.")
    parser.add_argument("--results-csv", default="outputs/logs/table1_main_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"No results CSV found at {csv_path}")

    with csv_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    print("Method | Dataset | Trainable Params | PSNR (higher) | SSIM (higher) | GPU Mem MB (lower)")
    print("--- | --- | ---: | ---: | ---: | ---:")
    for row in rows:
        print(
            f"{row['method']} | {row['dataset']} | {int(row['trainable_params']):,} | "
            f"{float(row['psnr']):.4f} | {float(row['ssim']):.4f} | {row['gpu_mem_mb']}"
        )


if __name__ == "__main__":
    main()
