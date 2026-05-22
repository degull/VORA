import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-style Table 3 SOTA comparison.")
    parser.add_argument("--input-csv", default="outputs/logs/table3_sota_comparison.csv")
    parser.add_argument("--output-csv", default="outputs/logs/table3_sota_comparison_summary.csv")
    parser.add_argument("--output-md", default="outputs/logs/table3_sota_comparison_summary.md")
    return parser.parse_args()


def metric_value(row: dict[str, str], prefix: str) -> tuple[float, float] | None:
    psnr = row.get(f"{prefix}_psnr", "")
    ssim = row.get(f"{prefix}_ssim", "")
    if not psnr or not ssim:
        return None
    return float(psnr), float(ssim)


def metric(row: dict[str, str], prefix: str) -> str:
    value = metric_value(row, prefix)
    if value is None:
        return "-"
    psnr, ssim = value
    return f"{psnr:.4f} / {ssim:.4f}"


def average_metric(row: dict[str, str]) -> str:
    values = [
        metric_value(row, "rain100h"),
        metric_value(row, "csd"),
        metric_value(row, "gopro"),
        metric_value(row, "reside6k"),
    ]
    present = [value for value in values if value is not None]
    if not present:
        return "-"
    avg_psnr = sum(value[0] for value in present) / len(present)
    avg_ssim = sum(value[1] for value in present) / len(present)
    return f"{avg_psnr:.4f} / {avg_ssim:.4f}"


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary = []
    for row in rows:
        summary.append(
            {
                "Model": row["model"],
                "Source": row["source"],
                "Params": row.get("params_m", "") or "-",
                "Rain100H": metric(row, "rain100h"),
                "CSD": metric(row, "csd"),
                "GoPro": metric(row, "gopro"),
                "RESIDE-6K": metric(row, "reside6k"),
                "Average": average_metric(row),
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
        "# Table 3. Comparison with SOTA Restoration Models",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"No Table 3 CSV found at {csv_path}. Run build_table3_sota_comparison.py first.")

    with csv_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    summary = build_summary(rows)
    write_csv(Path(args.output_csv), summary)
    write_markdown(Path(args.output_md), summary)

    for row in rows:
        print(
            f"{row['model']} | {row['source']} | {row.get('params_m', '-') or '-'} | "
            f"Rain100H={metric(row, 'rain100h')} | CSD={metric(row, 'csd')} | "
            f"GoPro={metric(row, 'gopro')} | RESIDE-6K={metric(row, 'reside6k')} | "
            f"Avg={average_metric(row)}"
        )
    print("")
    print(f"Saved CSV to {args.output_csv}")
    print(f"Saved Markdown to {args.output_md}")


if __name__ == "__main__":
    main()
