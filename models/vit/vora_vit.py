from __future__ import annotations

from dataclasses import dataclass

import timm
from torch import nn

from models.vora_linear import VoRALinear


@dataclass
class ReplacementStats:
    replaced: int = 0
    skipped: int = 0


@dataclass
class VoRAViTBuildResult:
    model: nn.Module
    stats: ReplacementStats
    target_keywords: tuple[str, ...]


VIT_TARGET_PRESETS: dict[str, tuple[str, ...]] = {
    "mlp": ("fc1", "fc2"),
    "attn": ("qkv", "proj"),
    "all": ("qkv", "proj", "fc1", "fc2"),
}


def _copy_linear_weights(source: nn.Linear, target: VoRALinear) -> None:
    target.linear.weight.data.copy_(source.weight.data)
    if source.bias is not None and target.linear.bias is not None:
        target.linear.bias.data.copy_(source.bias.data)


def _should_replace(name: str, target_keywords: tuple[str, ...]) -> bool:
    return any(keyword in name for keyword in target_keywords)


def replace_linear_with_vora(
    module: nn.Module,
    target_keywords: tuple[str, ...] = ("mlp.fc1", "mlp.fc2"),
    lora_rank: int = 4,
    volterra_rank: int = 4,
    freeze_base: bool = True,
) -> ReplacementStats:
    """Replace selected nn.Linear layers in a module tree with VoRALinear.

    The default targets ViT MLP projections first because they are the least
    invasive place to validate a Linear PEFT adapter inside a real backbone.
    """

    stats = ReplacementStats()

    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if _should_replace(child_name, target_keywords):
                replacement = VoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    lora_rank=lora_rank,
                    volterra_rank=volterra_rank,
                    bias=child.bias is not None,
                    freeze_base=freeze_base,
                )
                _copy_linear_weights(child, replacement)
                setattr(module, child_name, replacement)
                stats.replaced += 1
            else:
                stats.skipped += 1
            continue

        child_stats = replace_linear_with_vora(
            child,
            target_keywords=target_keywords,
            lora_rank=lora_rank,
            volterra_rank=volterra_rank,
            freeze_base=freeze_base,
        )
        stats.replaced += child_stats.replaced
        stats.skipped += child_stats.skipped

    return stats


def get_vit_target_keywords(target: str) -> tuple[str, ...]:
    if target not in VIT_TARGET_PRESETS:
        choices = ", ".join(sorted(VIT_TARGET_PRESETS))
        raise ValueError(f"Unknown target '{target}'. Available targets: {choices}.")
    return VIT_TARGET_PRESETS[target]


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def build_vora_vit(
    model_name: str = "vit_tiny_patch16_224",
    target: str = "mlp",
    lora_rank: int = 4,
    volterra_rank: int = 4,
    pretrained: bool = False,
    freeze_backbone: bool = True,
    num_classes: int | None = None,
) -> VoRAViTBuildResult:
    """Build a timm ViT backbone and insert VoRA adapters into selected Linear layers."""

    create_kwargs = {"pretrained": pretrained}
    if num_classes is not None:
        create_kwargs["num_classes"] = num_classes

    model = timm.create_model(model_name, **create_kwargs)

    if freeze_backbone:
        freeze_module(model)

    target_keywords = get_vit_target_keywords(target)
    stats = replace_linear_with_vora(
        model,
        target_keywords=target_keywords,
        lora_rank=lora_rank,
        volterra_rank=volterra_rank,
        freeze_base=freeze_backbone,
    )

    return VoRAViTBuildResult(
        model=model,
        stats=stats,
        target_keywords=target_keywords,
    )
