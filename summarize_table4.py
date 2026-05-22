import argparse
import csv
from pathlib import Path


DISPLAY_BACKBONES = [
    ("swinir_official", "SwinIR"),
    ("uformer", "Uformer"),
    ("hat", "HAT"),
    ("mambairv2", "MambaIRv2"),
    ("adair", "AdaIR"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print Table 4 backbone generalization.")
    parser.add_argument("--results-csv", default="outputs/logs/table4_backbone_generalization.csv")
    return parser.parse_args()


def metric(rows: list[dict[str, str]], backbone: str, dataset: str, method: str) -> str:
    for row in rows:
        if row["backbone"] == backbone and row["dataset"] == dataset and row["method"] == method:
            return f"{float(row['psnr']):.4f}/{float(row['ssim']):.4f}"
    return "-"


def params(rows: list[dict[str, str]], backbone: str, method: str) -> str:
    for row in rows:
        if row["backbone"] == backbone and row["method"] == method:
            return f"{int(row['trainable_params']):,}"
    return "-"


def main() -> None:
    args = parse_args()
    csv_path = Path(args.results_csv)
    rows = []
    if csv_path.exists():
        with csv_path.open("r", newline="") as handle:
            rows = list(csv.DictReader(handle))

    print("Table 4. Backbone Generalization")
    print("")
    print(
        "Backbone | LoRA Params | VoRA-v1 Params | "
        "Rain100H LoRA | Rain100H VoRA-v1 | GoPro LoRA | GoPro VoRA-v1"
    )
    print("--- | ---: | ---: | ---: | ---: | ---: | ---:")
    for backbone_key, label in DISPLAY_BACKBONES:
        print(
            f"{label} | "
            f"{params(rows, backbone_key, 'lora')} | "
            f"{params(rows, backbone_key, 'vora_v1')} | "
            f"{metric(rows, backbone_key, 'rain100h', 'lora')} | "
            f"{metric(rows, backbone_key, 'rain100h', 'vora_v1')} | "
            f"{metric(rows, backbone_key, 'gopro', 'lora')} | "
            f"{metric(rows, backbone_key, 'gopro', 'vora_v1')}"
        )


if __name__ == "__main__":
    main()
