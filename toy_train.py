import torch
from torch import nn

from config import ExperimentConfig, ModelConfig
from models import LoRALinear, VoRALinear
from utils.model_utils import count_parameters


def make_target(x: torch.Tensor, out_features: int) -> torch.Tensor:
    """Synthetic target with linear and quadratic feature interactions."""
    in_features = x.shape[-1]
    linear_weight = torch.randn(in_features, out_features, device=x.device) * 0.02
    quad_weight = torch.randn(in_features, out_features, device=x.device) * 0.01
    return x @ linear_weight + (x * x) @ quad_weight


def train_model(name: str, model: nn.Module, x: torch.Tensor, target: torch.Tensor, steps: int = 200) -> None:
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-2)
    loss_fn = nn.MSELoss()

    print(f"{name}")
    print(f"  trainable params: {count_parameters(model, trainable_only=True):,}")

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)

        if isinstance(model, VoRALinear):
            pred, branches = model(x, return_branches=True)
        else:
            pred = model(x)
            branches = {}

        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"  step {step:03d} | loss {loss.item():.6f}")
            if branches:
                print(
                    "           "
                    f"lora norm {branches['lora'].norm().item():.6f} | "
                    f"volterra norm {branches['volterra'].norm().item():.6f}"
                )


def main() -> None:
    model_cfg = ModelConfig(in_features=64, out_features=64, lora_rank=4, volterra_rank=4)
    exp_cfg = ExperimentConfig(batch_size=8, num_tokens=32, seed=42)
    torch.manual_seed(exp_cfg.seed)

    x = torch.randn(exp_cfg.batch_size, exp_cfg.num_tokens, model_cfg.in_features)
    target = make_target(x, model_cfg.out_features)

    lora = LoRALinear(
        in_features=model_cfg.in_features,
        out_features=model_cfg.out_features,
        rank=model_cfg.lora_rank,
        alpha=model_cfg.lora_alpha,
        freeze_base=model_cfg.freeze_base,
    )
    vora = VoRALinear(
        in_features=model_cfg.in_features,
        out_features=model_cfg.out_features,
        lora_rank=model_cfg.lora_rank,
        volterra_rank=model_cfg.volterra_rank,
        lora_alpha=model_cfg.lora_alpha,
        volterra_alpha=model_cfg.volterra_alpha,
        freeze_base=model_cfg.freeze_base,
    )

    train_model("LoRALinear toy fit", lora, x, target)
    train_model("VoRALinear toy fit", vora, x, target)


if __name__ == "__main__":
    main()
