import argparse
import csv
from pathlib import Path


VARIANTS = [
    ("frozen", "Frozen Backbone", "No", "No", "No"),
    ("lora", "LoRA only", "Yes", "No", "No"),
    ("volterra_only", "Volterra only", "No", "Yes", "Yes"),
    ("lora_linear_volterra", "LoRA + Linear Volterra", "Yes", "Yes", "No"),
    ("vora_v1", "VoRA-v1", "Yes", "Yes", "Yes"),
    ("full_ft", "Full Fine-tuning", "-", "-", "-"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Table 6 ablation results.")
    parser.add_argument("--input-csv", default="outputs/logs/table6_ablation_study.csv")
    parser.add_argument("--output-csv", default="outputs/logs/table6_ablation_study_summary.csv")
    parser.add_argument("--output-md", default="outputs/logs/table6_ablation_study_summary.md")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def latest_by_method(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_method = {}
    for row in rows:
        by_method[row["method"]] = row
    return by_method


def fmt_params(row: dict[str, str] | None) -> str:
    if row is None:
        return "TBD"
    return f"{int(row['trainable_params']):,}"


def fmt_metric(row: dict[str, str] | None, key: str) -> str:
    if row is None:
        return "TBD"
    return f"{float(row[key]):.4f}"


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_method = latest_by_method(rows)
    summary = []
    for method, variant, lora_branch, volterra_branch, quadratic in VARIANTS:
        row = by_method.get(method)
        if row is None and method == "full_ft":
            continue
        summary.append(
            {
                "Variant": variant,
                "LoRA Branch": lora_branch,
                "Volterra Branch": volterra_branch,
                "Quadratic Interaction": quadratic,
                "Trainable Params": fmt_params(row),
                "PSNR": fmt_metric(row, "psnr"),
                "SSIM": fmt_metric(row, "ssim"),
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
        "# Table 6. Ablation Study of VoRA Components",
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
    if not summary:
        print("No Table 6 results found yet.")
        return

    write_csv(Path(args.output_csv), summary)
    write_markdown(Path(args.output_md), summary)

    for row in summary:
        print(
            f"{row['Variant']} | LoRA={row['LoRA Branch']} | "
            f"Volterra={row['Volterra Branch']} | Quadratic={row['Quadratic Interaction']} | "
            f"params={row['Trainable Params']} | PSNR={row['PSNR']} | SSIM={row['SSIM']}"
        )
    print("")
    print(f"Saved CSV to {args.output_csv}")
    print(f"Saved Markdown to {args.output_md}")


if __name__ == "__main__":
    main()
