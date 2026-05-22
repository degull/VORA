import argparse
import csv
from pathlib import Path


DATASETS = ["rain100h", "csd", "gopro", "reside6k"]
DATASET_LABELS = {
    "rain100h": "Rain100H",
    "csd": "CSD",
    "gopro": "GoPro",
    "reside6k": "RESIDE-6K",
}
METHOD_LABELS = {
    "lora": "LoRA",
    "vora_v1": "VoRA-v1",
    "vora_token": "VoRA-token",
    "vora_full": "VoRA-full",
}
BACKBONE_LABELS = {
    "swinir_official": "SwinIR",
    "swinir_lite": "SwinIR-lite",
    "uformer": "Uformer",
    "hat": "HAT",
    "adair": "AdaIR",
    "mambairv2": "MambaIRv2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final SOTA + adapter comparison table.")
    parser.add_argument("--sota-csv", default="outputs/logs/table3_sota_comparison.csv")
    parser.add_argument(
        "--adapter-csvs",
        nargs="*",
        default=[
            "outputs/logs/table5_param_matched_comparison.csv",
            "outputs/logs/table4_backbone_generalization.csv",
            "outputs/logs/table2_task_generalization.csv",
        ],
    )
    parser.add_argument("--output-csv", default="outputs/logs/final_sota_adapter_comparison.csv")
    parser.add_argument("--output-md", default="outputs/logs/final_sota_adapter_comparison.md")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_from_sota(row: dict[str, str], dataset: str) -> str:
    psnr = row.get(f"{dataset}_psnr", "")
    ssim = row.get(f"{dataset}_ssim", "")
    if not psnr or not ssim:
        return "-"
    return f"{float(psnr):.4f} / {float(ssim):.4f}"


def metric_from_adapter(row: dict[str, str]) -> str:
    return f"{float(row['psnr']):.4f} / {float(row['ssim']):.4f}"


def average_from_cells(cells: list[str]) -> str:
    values = []
    for cell in cells:
        if cell == "-":
            continue
        psnr_text, ssim_text = [part.strip() for part in cell.split("/")]
        values.append((float(psnr_text), float(ssim_text)))
    if not values:
        return "-"
    return f"{sum(v[0] for v in values) / len(values):.4f} / {sum(v[1] for v in values) / len(values):.4f}"


def build_sota_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        cells = [metric_from_sota(row, dataset) for dataset in DATASETS]
        output.append(
            {
                "Group": "Reported full model",
                "Model / Backbone": row["model"],
                "Adapter": "-",
                "Rank": "-",
                "Trainable Params": row.get("params_m", "") or "-",
                "Rain100H": cells[0],
                "CSD": cells[1],
                "GoPro": cells[2],
                "RESIDE-6K": cells[3],
                "Average": average_from_cells(cells),
                "Source": row.get("source", "-"),
            }
        )
    return output


def row_priority(row: dict[str, str]) -> tuple[int, int]:
    method_order = {"lora": 0, "vora_v1": 1, "vora_token": 2, "vora_full": 3}
    rank = int(row.get("rank") or 0)
    method = row.get("method", "")
    return method_order.get(method, 99), rank


def build_adapter_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        dataset = row.get("dataset", "")
        method = row.get("method", "")
        backbone = row.get("backbone", "")
        rank = row.get("rank", "")
        if dataset not in DATASETS or method not in METHOD_LABELS or not backbone or not rank:
            continue
        key = (backbone, method, rank, row.get("trainable_params", ""))
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = {
                "Group": "Adapter on backbone",
                "Model / Backbone": BACKBONE_LABELS.get(backbone, backbone),
                "Adapter": METHOD_LABELS[method],
                "Rank": rank,
                "Trainable Params": f"{int(row['trainable_params']):,}",
                "Rain100H": "-",
                "CSD": "-",
                "GoPro": "-",
                "RESIDE-6K": "-",
                "Average": "-",
                "Source": "ours",
                "_method": method,
            }
            existing = grouped[key]
        existing[DATASET_LABELS[dataset]] = metric_from_adapter(row)

    output = []
    for row in grouped.values():
        cells = [row[DATASET_LABELS[dataset]] for dataset in DATASETS]
        row["Average"] = average_from_cells(cells)
        row.pop("_method", None)
        output.append(row)

    return sorted(output, key=lambda row: (row["Model / Backbone"], row["Adapter"], int(row["Rank"])))


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
        "# Final Table. SOTA Restoration Models and Adapter Comparisons",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sota_rows = build_sota_rows(read_csv(Path(args.sota_csv)))

    adapter_input_rows = []
    for csv_path in args.adapter_csvs:
        adapter_input_rows.extend(read_csv(Path(csv_path)))
    adapter_rows = build_adapter_rows(adapter_input_rows)

    rows = sota_rows + adapter_rows
    if not rows:
        raise RuntimeError("No rows found. Check input CSV paths.")

    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), rows)

    print(f"Saved CSV to {args.output_csv}")
    print(f"Saved Markdown to {args.output_md}")
    print(f"Rows: {len(rows)} ({len(sota_rows)} reported full-model rows, {len(adapter_rows)} adapter rows)")


if __name__ == "__main__":
    main()
