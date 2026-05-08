from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from models.lora_linear import LoRALinear
from models.vora_full_linear import VoRAFullLinear, VoRATokenLinear
from models.vora_linear import VoRALinear


@dataclass
class AdapterStats:
    replaced: int = 0
    skipped: int = 0


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _copy_linear(source: nn.Linear, target: LoRALinear | VoRALinear | VoRATokenLinear | VoRAFullLinear) -> None:
    target.linear.weight.data.copy_(source.weight.data)
    if source.bias is not None and target.linear.bias is not None:
        target.linear.bias.data.copy_(source.bias.data)


def replace_linear_adapters(
    module: nn.Module,
    method: str,
    target_keywords: tuple[str, ...] = ("qkv", "proj", "mlp"),
    rank: int = 4,
    prefix: str = "",
) -> AdapterStats:
    stats = AdapterStats()
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if any(keyword in full_name for keyword in target_keywords):
                if method == "lora":
                    replacement = LoRALinear(child.in_features, child.out_features, rank=rank, bias=child.bias is not None)
                elif method == "vora_v1":
                    replacement = VoRALinear(
                        child.in_features,
                        child.out_features,
                        lora_rank=rank,
                        volterra_rank=rank,
                        bias=child.bias is not None,
                    )
                elif method == "vora_token":
                    replacement = VoRATokenLinear(
                        child.in_features,
                        child.out_features,
                        lora_rank=rank,
                        volterra_rank=rank,
                        bias=child.bias is not None,
                    )
                elif method == "vora_full":
                    replacement = VoRAFullLinear(
                        child.in_features,
                        child.out_features,
                        lora_rank=rank,
                        volterra_rank=rank,
                        num_experts=4,
                        bias=child.bias is not None,
                    )
                else:
                    raise ValueError(f"Unsupported adapter method: {method}")
                _copy_linear(child, replacement)
                setattr(module, name, replacement)
                stats.replaced += 1
            else:
                stats.skipped += 1
            continue

        child_stats = replace_linear_adapters(
            child,
            method,
            target_keywords=target_keywords,
            rank=rank,
            prefix=full_name,
        )
        stats.replaced += child_stats.replaced
        stats.skipped += child_stats.skipped
    return stats
