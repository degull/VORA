import math

import torch
from torch import nn

from .volterra import QuadraticVolterra


class VolterraOnlyLinear(nn.Module):
    """Frozen base linear layer with only a quadratic Volterra adapter branch."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.volterra = QuadraticVolterra(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        if freeze_base:
            self.freeze_base()

    def freeze_base(self) -> None:
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.volterra(x)


class LinearVolterraLinear(nn.Module):
    """LoRA plus a second low-rank linear branch without quadratic interaction."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int = 8,
        volterra_rank: int = 8,
        lora_alpha: float = 1.0,
        volterra_alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if lora_rank <= 0 or volterra_rank <= 0:
            raise ValueError("adapter ranks must be positive.")

        self.lora_scaling = lora_alpha / lora_rank
        self.volterra_scaling = volterra_alpha / volterra_rank
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.A_l = nn.Linear(in_features, lora_rank, bias=False)
        self.B_l = nn.Linear(lora_rank, out_features, bias=False)
        self.A_v = nn.Linear(in_features, volterra_rank, bias=False)
        self.B_v = nn.Linear(volterra_rank, out_features, bias=False)
        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)
        nn.init.kaiming_uniform_(self.A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_v.weight)

    def freeze_base(self) -> None:
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        dropped = self.dropout(x)
        lora = self.B_l(self.A_l(dropped)) * self.lora_scaling
        linear_volterra = self.B_v(self.A_v(dropped)) * self.volterra_scaling
        return base + lora + linear_volterra
