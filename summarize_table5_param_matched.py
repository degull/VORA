import argparse
import csv
from pathlib import Path


TASKS = {
    "rain100h": ("Deraining", "Rain100H"),
    "csd": ("Desnowing", "CSD"),
    "gopro": ("Deblurring", "GoPro"),
    "reside6k": ("Dehazing", "RESIDE-6K"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize parameter-matched LoRA vs VoRA results.")
    parser.add_argument("--input-csv", default="outputs/logs/table5_param_matched_comparison.csv")
    parser.add_argument("--output-csv", default="outputs/logs/table5_param_matched_summary.csv")
    parser.add_argument("--output-md", default="outputs/logs/table5_param_matched_summary.md")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key = {
        (row["dataset"], row["method"], row["rank"]): row
        for row in rows
    }
    summary = []
    for dataset, (task, display_dataset) in TASKS.items():
        baseline = by_key.get((dataset, "lora", "8"))
        vora = by_key.get((dataset, "vora_v1", "4"))

        for row in (baseline, vora):
            if row is None:
                method = "LoRA" if len(summary) % 2 == 0 else "VoRA-v1"
                rank = "8" if method == "LoRA" else "4"
                summary.append(
                    {
                        "Task": task,
                        "Dataset": display_dataset,
                        "Method": method,
                        "Rank": rank,
                        "Trainable Params": "...",
                        "PSNR ↑": "...",
                        "SSIM ↑": "...",
                        "ΔPSNR": "-",
                        "ΔSSIM": "-",
                    }
                )
                continue

            is_lora = row["method"] == "lora"
            psnr = float(row["psnr"])
            ssim = float(row["ssim"])
            delta_psnr = None
            delta_ssim = None
            if not is_lora and baseline is not None:
                delta_psnr = psnr - float(baseline["psnr"])
                delta_ssim = ssim - float(baseline["ssim"])

            summary.append(
                {
                    "Task": task,
                    "Dataset": display_dataset,
                    "Method": "LoRA" if is_lora else "VoRA-v1",
                    "Rank": row["rank"],
                    "Trainable Params": f"{int(row['trainable_params']):,}",
                    "PSNR ↑": f"{psnr:.4f}",
                    "SSIM ↑": f"{ssim:.4f}",
                    "ΔPSNR": fmt_delta(delta_psnr),
                    "ΔSSIM": fmt_delta(delta_ssim),
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
        "# Table X. Parameter-Matched Comparison Across Restoration Tasks",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.input_csv))
    summary = build_summary(rows)
    write_csv(Path(args.output_csv), summary)
    write_markdown(Path(args.output_md), summary)

    for row in summary:
        print(
            f"{row['Task']} | {row['Dataset']} | {row['Method']} | "
            f"r={row['Rank']} | params={row['Trainable Params']} | "
            f"PSNR={row['PSNR ↑']} | SSIM={row['SSIM ↑']} | "
            f"dPSNR={row['ΔPSNR']} | dSSIM={row['ΔSSIM']}"
        )
    print("")
    print(f"Saved CSV to {args.output_csv}")
    print(f"Saved Markdown to {args.output_md}")


if __name__ == "__main__":
    main()
