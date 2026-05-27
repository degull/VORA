import argparse
import csv
from pathlib import Path


METHODS = [
    ("lora", "LoRA", "8"),
    ("vora_v1", "VoRA-v1", "4"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize degradation interaction results.")
    parser.add_argument("--input-csv", default="outputs/logs/table5_degradation_interaction.csv")
    parser.add_argument("--output-csv", default="outputs/logs/table5_degradation_interaction_summary.csv")
    parser.add_argument("--output-md", default="outputs/logs/table5_degradation_interaction_summary.md")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def intensity_key(row: dict[str, str]) -> float:
    return float(row.get("degradation_intensity", "-1"))


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key = {
        (float(row.get("degradation_intensity", "-1")), row["method"], row["rank"]): row
        for row in rows
    }
    intensities = sorted({intensity_key(row) for row in rows if intensity_key(row) >= 0})
    summary = []
    for intensity in intensities:
        lora = by_key.get((intensity, "lora", "8"))
        for method, method_label, rank in METHODS:
            row = by_key.get((intensity, method, rank))
            if row is None:
                summary.append(
                    {
                        "Base Dataset": "GoPro",
                        "Composite": "Blur+Noise",
                        "Noise σ": f"{intensity:.2f}",
                        "Method": method_label,
                        "Rank": rank,
                        "Params": "TBD",
                        "PSNR": "TBD",
                        "SSIM": "TBD",
                        "ΔPSNR vs LoRA": "-" if method == "lora" else "TBD",
                        "ΔSSIM vs LoRA": "-" if method == "lora" else "TBD",
                    }
                )
                continue

            psnr = float(row["psnr"])
            ssim = float(row["ssim"])
            delta_psnr = "-"
            delta_ssim = "-"
            if method != "lora" and lora is not None:
                delta_psnr = f"{psnr - float(lora['psnr']):+.4f}"
                delta_ssim = f"{ssim - float(lora['ssim']):+.4f}"

            summary.append(
                {
                    "Base Dataset": "GoPro",
                    "Composite": "Blur+Noise",
                    "Noise σ": f"{intensity:.2f}",
                    "Method": method_label,
                    "Rank": rank,
                    "Params": f"{int(row['trainable_params']):,}",
                    "PSNR": f"{psnr:.4f}",
                    "SSIM": f"{ssim:.4f}",
                    "ΔPSNR vs LoRA": delta_psnr,
                    "ΔSSIM vs LoRA": delta_ssim,
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
        "# Table 5. Degradation Interaction under Increasing Noise Intensity",
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
    if not summary:
        print("No Table 5 results found yet.")
        return
    write_csv(Path(args.output_csv), summary)
    write_markdown(Path(args.output_md), summary)

    for row in summary:
        print(
            f"{row['Composite']} | noise={row['Noise σ']} | {row['Method']} | "
            f"r={row['Rank']} | params={row['Params']} | PSNR={row['PSNR']} | "
            f"SSIM={row['SSIM']} | dPSNR={row['ΔPSNR vs LoRA']} | dSSIM={row['ΔSSIM vs LoRA']}"
        )
    print("")
    print(f"Saved CSV to {args.output_csv}")
    print(f"Saved Markdown to {args.output_md}")


if __name__ == "__main__":
    main()
