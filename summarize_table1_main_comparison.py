import argparse
import csv
from pathlib import Path


METHOD_ORDER = ["full_ft", "frozen", "lora", "vora_v1"]
METHOD_LABELS = {
    "full_ft": "Full fine-tuning",
    "frozen": "Frozen",
    "lora": "LoRA",
    "vora_v1": "VoRA-v1",
}
DATASET_ORDER = ["rain100h", "csd", "gopro", "reside6k"]
DATASET_LABELS = {
    "rain100h": "Rain100H",
    "csd": "CSD",
    "gopro": "GoPro",
    "reside6k": "RESIDE-6K",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-style Table 1 from main comparison results.")
    parser.add_argument("--input-csv", default="outputs/logs/table1_main_comparison.csv")
    parser.add_argument("--output-csv", default="outputs/logs/table1_main_comparison_summary.csv")
    parser.add_argument("--output-md", default="outputs/logs/table1_main_comparison_summary.md")
    return parser.parse_args()


def valid_row(row: dict[str, str]) -> bool:
    return (
        row.get("method") in METHOD_ORDER
        and row.get("dataset") in DATASET_ORDER
        and row.get("psnr", "").replace(".", "", 1).replace("-", "", 1).isdigit()
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [row for row in csv.DictReader(handle) if valid_row(row)]


def metric_cell(row: dict[str, str] | None) -> str:
    if row is None:
        return "-"
    return f"{float(row['psnr']):.4f} / {float(row['ssim']):.4f}"


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key = {(row["method"], row["dataset"]): row for row in rows}
    summary = []
    for method in METHOD_ORDER:
        method_rows = [by_key.get((method, dataset)) for dataset in DATASET_ORDER]
        present_rows = [row for row in method_rows if row is not None]
        if not present_rows:
            continue

        avg_psnr = sum(float(row["psnr"]) for row in present_rows) / len(present_rows)
        avg_ssim = sum(float(row["ssim"]) for row in present_rows) / len(present_rows)
        params = int(present_rows[0]["trainable_params"])
        gpu_mem = float(present_rows[0]["gpu_mem_mb"])

        summary.append(
            {
                "Method": METHOD_LABELS[method],
                "Trainable Params": f"{params:,}",
                "Rain100H PSNR/SSIM": metric_cell(by_key.get((method, "rain100h"))),
                "CSD PSNR/SSIM": metric_cell(by_key.get((method, "csd"))),
                "GoPro PSNR/SSIM": metric_cell(by_key.get((method, "gopro"))),
                "RESIDE-6K PSNR/SSIM": metric_cell(by_key.get((method, "reside6k"))),
                "Avg. PSNR/SSIM": f"{avg_psnr:.4f} / {avg_ssim:.4f}",
                "GPU Mem. (MB)": f"{gpu_mem:.2f}",
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0])
    lines = [
        "# Table 1. Main Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = build_summary(read_rows(Path(args.input_csv)))
    write_csv(Path(args.output_csv), summary)
    write_markdown(Path(args.output_md), summary)

    for row in summary:
        print(
            f"{row['Method']} | params={row['Trainable Params']} | "
            f"Rain100H={row['Rain100H PSNR/SSIM']} | CSD={row['CSD PSNR/SSIM']} | "
            f"GoPro={row['GoPro PSNR/SSIM']} | RESIDE-6K={row['RESIDE-6K PSNR/SSIM']} | "
            f"Avg={row['Avg. PSNR/SSIM']}"
        )
    print("")
    print(f"Saved CSV to {args.output_csv}")
    print(f"Saved Markdown to {args.output_md}")


if __name__ == "__main__":
    main()
