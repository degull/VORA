import math

import torch
from torch import nn


class LoRAConv2d(nn.Module):
    """LoRA adapter for 1x1 Conv2d projections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int = 8,
        alpha: float = 1.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.A_l = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.B_l = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)

    def freeze_base(self) -> None:
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.B_l(self.A_l(x)) * self.scaling
