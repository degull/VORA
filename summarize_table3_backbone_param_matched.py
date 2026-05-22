import argparse
import csv
from pathlib import Path


BACKBONES = [
    ("swinir_official", "SwinIR"),
    ("uformer", "Uformer"),
    ("hat", "HAT"),
    ("adair", "AdaIR"),
    ("dfpir_restormer", "DFPIR/Restormer"),
    ("mambairv2", "MambaIRv2"),
]
METHODS = [
    ("lora", "LoRA", "8"),
    ("vora_v1", "VoRA-v1", "4"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize backbone-wise parameter-matched adapter comparisons.")
    parser.add_argument("--dataset", default="rain100h")
    parser.add_argument(
        "--input-csvs",
        nargs="*",
        default=[
            "outputs/logs/table5_param_matched_comparison.csv",
            "outputs/logs/table3_backbone_param_matched.csv",
        ],
    )
    parser.add_argument("--output-csv", default="outputs/logs/table3_backbone_param_matched_summary.csv")
    parser.add_argument("--output-md", default="outputs/logs/table3_backbone_param_matched_summary.md")
    return parser.parse_args()


def read_rows(paths: list[str]) -> list[dict[str, str]]:
    rows = []
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            continue
        with path.open("r", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("backbone", ""),
        row.get("dataset", ""),
        row.get("method", ""),
        row.get("rank", ""),
    )


def build_summary(rows: list[dict[str, str]], dataset: str) -> list[dict[str, str]]:
    by_key = {row_key(row): row for row in rows}
    summary = []
    for backbone, backbone_label in BACKBONES:
        lora = by_key.get((backbone, dataset, "lora", "8"))
        for method, method_label, rank in METHODS:
            row = by_key.get((backbone, dataset, method, rank))
            if row is None:
                summary.append(
                    {
                        "Backbone": backbone_label,
                        "Dataset": "Rain100H" if dataset == "rain100h" else dataset,
                        "Adapter": method_label,
                        "Rank": rank,
                        "Trainable Params": "TBD",
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
                    "Backbone": backbone_label,
                    "Dataset": "Rain100H" if dataset == "rain100h" else dataset,
                    "Adapter": method_label,
                    "Rank": rank,
                    "Trainable Params": f"{int(row['trainable_params']):,}",
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
        "# Table 3. Backbone-wise Parameter-Matched Adapter Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = build_summary(read_rows(args.input_csvs), args.dataset)
    write_csv(Path(args.output_csv), summary)
    write_markdown(Path(args.output_md), summary)

    for row in summary:
        print(
            f"{row['Backbone']} | {row['Dataset']} | {row['Adapter']} | r={row['Rank']} | "
            f"params={row['Trainable Params']} | PSNR={row['PSNR']} | SSIM={row['SSIM']} | "
            f"dPSNR={row['ΔPSNR vs LoRA']} | dSSIM={row['ΔSSIM vs LoRA']}"
        )
    print("")
    print(f"Saved CSV to {args.output_csv}")
    print(f"Saved Markdown to {args.output_md}")


if __name__ == "__main__":
    main()
