import argparse
import csv
from pathlib import Path


TASK_BY_DATASET = {
    "rain100h": "Deraining",
    "csd": "Desnowing",
    "sidd": "Denoising",
    "gopro": "Deblurring",
    "reside6k": "Dehazing",
}

METHOD_LABELS = {
    "lora": "LoRA",
    "vora_v1": "VoRA-v1",
    "vora_token": "VoRA-token",
    "vora_full": "VoRA-full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print Table 2 from task-generalization CSV.")
    parser.add_argument("--results-csv", default="outputs/logs/table2_task_generalization.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"No Table 2 CSV found at {csv_path}")

    with csv_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    table = {}
    for row in rows:
        method = row["method"]
        task = TASK_BY_DATASET.get(row["dataset"], row["dataset"])
        table.setdefault(method, {})[task] = f"{float(row['psnr']):.4f}/{float(row['ssim']):.4f}"

    print(
        "Method | Deraining PSNR/SSIM | Desnowing PSNR/SSIM | "
        "Denoising PSNR/SSIM | Deblurring PSNR/SSIM | Dehazing PSNR/SSIM"
    )
    print("--- | ---: | ---: | ---: | ---: | ---:")
    for method in ["lora", "vora_v1", "vora_token", "vora_full"]:
        values = table.get(method, {})
        print(
            f"{METHOD_LABELS[method]} | "
            f"{values.get('Deraining', '-')} | "
            f"{values.get('Desnowing', '-')} | "
            f"{values.get('Denoising', '-')} | "
            f"{values.get('Deblurring', '-')} | "
            f"{values.get('Dehazing', '-')}"
        )


if __name__ == "__main__":
    main()
