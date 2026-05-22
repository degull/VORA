import argparse
import csv
from pathlib import Path


DATASET_COLUMNS = {
    "rain100h": ("rain100h_psnr", "rain100h_ssim"),
    "csd": ("csd_psnr", "csd_ssim"),
    "gopro": ("gopro_psnr", "gopro_ssim"),
    "reside6k": ("reside6k_psnr", "reside6k_ssim"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Table 3 SOTA comparison CSV.")
    parser.add_argument("--baseline-csv", default="papers/ablations/table3_sota_baselines.csv")
    parser.add_argument("--table2-csv", default="outputs/logs/table2_task_generalization.csv")
    parser.add_argument("--output-csv", default="outputs/logs/table3_sota_comparison.csv")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def build_ours_row(table2_rows: list[dict[str, str]]) -> dict[str, str]:
    row = {
        "model": "Ours: SwinIR + VoRA-full",
        "source": "ours",
        "params_m": "",
        "rain100h_psnr": "",
        "rain100h_ssim": "",
        "csd_psnr": "",
        "csd_ssim": "",
        "gopro_psnr": "",
        "gopro_ssim": "",
        "reside6k_psnr": "",
        "reside6k_ssim": "",
        "notes": "SwinIR official base + VoRA-full, rank=4",
    }

    for result in table2_rows:
        if result["method"] != "vora_full":
            continue
        dataset = result["dataset"]
        if dataset not in DATASET_COLUMNS:
            continue
        psnr_col, ssim_col = DATASET_COLUMNS[dataset]
        row[psnr_col] = f"{float(result['psnr']):.4f}"
        row[ssim_col] = f"{float(result['ssim']):.4f}"
        if result.get("trainable_params"):
            row["params_m"] = f"{int(result['trainable_params']) / 1_000_000:.3f} trainable"
    return row


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline_csv)
    table2_path = Path(args.table2_csv)
    output_path = Path(args.output_csv)

    baseline_rows = read_rows(baseline_path)
    table2_rows = read_rows(table2_path)
    ours_row = build_ours_row(table2_rows)

    rows = [row for row in baseline_rows if row.get("model") != ours_row["model"]]
    rows.append(ours_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "source",
        "params_m",
        "rain100h_psnr",
        "rain100h_ssim",
        "csd_psnr",
        "csd_ssim",
        "gopro_psnr",
        "gopro_ssim",
        "reside6k_psnr",
        "reside6k_ssim",
        "notes",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved Table 3 CSV to {output_path}")


if __name__ == "__main__":
    main()
