import argparse

import torch

from models.vit.vora_vit import build_vora_vit
from utils.model_utils import count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test VoRA insertion into a timm ViT backbone.")
    parser.add_argument("--model", default="vit_tiny_patch16_224")
    parser.add_argument("--target", default="mlp", choices=("mlp", "attn", "all"))
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--volterra-rank", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    build = build_vora_vit(
        model_name=args.model,
        target=args.target,
        lora_rank=args.lora_rank,
        volterra_rank=args.volterra_rank,
        pretrained=False,
        freeze_backbone=True,
    )
    model = build.model
    model.eval()

    after_trainable = count_parameters(model, trainable_only=True)

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    print(f"Backbone: {args.model}")
    print(f"Target: {args.target}")
    print(f"Target keywords: {build.target_keywords}")
    print(f"Replaced Linear layers: {build.stats.replaced}")
    print(f"Skipped Linear layers:  {build.stats.skipped}")
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Trainable params after:  {after_trainable:,}")
    print("Forward success")


if __name__ == "__main__":
    main()
