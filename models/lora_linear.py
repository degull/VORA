import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Linear layer with a trainable low-rank LoRA adapter."""

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
        if rank <= 0:
            raise ValueError("rank must be positive.")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.A_l = nn.Linear(in_features, rank, bias=False)
        self.B_l = nn.Linear(rank, out_features, bias=False)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)

    def freeze_base(self) -> None:
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = self.B_l(self.A_l(self.dropout(x))) * self.scaling
        return base + lora
