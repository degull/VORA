import torch

from config import ExperimentConfig, ModelConfig
from models import LoRALinear, VoRALinear
from utils.model_utils import count_parameters


def describe(name: str, model: torch.nn.Module, x: torch.Tensor, inspect_branches: bool = False) -> None:
    with torch.no_grad():
        if inspect_branches:
            y, branches = model(x, return_branches=True)
        else:
            y = model(x)
            branches = {}

    print(f"{name}")
    print(f"  input shape:  {tuple(x.shape)}")
    print(f"  output shape: {tuple(y.shape)}")
    for branch_name, branch_output in branches.items():
        print(f"  {branch_name} norm: {branch_output.norm().item():.6f}")
    if branches:
        print(f"  output norm: {y.norm().item():.6f}")
    print(f"  total params: {count_parameters(model):,}")
    print(f"  trainable:    {count_parameters(model, trainable_only=True):,}")


def main() -> None:
    model_cfg = ModelConfig()
    exp_cfg = ExperimentConfig()
    torch.manual_seed(exp_cfg.seed)

    x = torch.randn(exp_cfg.batch_size, exp_cfg.num_tokens, model_cfg.in_features)

    lora = LoRALinear(
        in_features=model_cfg.in_features,
        out_features=model_cfg.out_features,
        rank=model_cfg.lora_rank,
        alpha=model_cfg.lora_alpha,
        dropout=model_cfg.dropout,
        freeze_base=model_cfg.freeze_base,
    )
    vora = VoRALinear(
        in_features=model_cfg.in_features,
        out_features=model_cfg.out_features,
        lora_rank=model_cfg.lora_rank,
        volterra_rank=model_cfg.volterra_rank,
        lora_alpha=model_cfg.lora_alpha,
        volterra_alpha=model_cfg.volterra_alpha,
        dropout=model_cfg.dropout,
        freeze_base=model_cfg.freeze_base,
    )

    describe("LoRALinear", lora, x)
    describe("VoRALinear", vora, x, inspect_branches=True)


if __name__ == "__main__":
    main()
