import math

import torch
from torch import nn


class VolterraOnlyConv2d(nn.Module):
    """Frozen 1x1 convolution with only a quadratic Volterra adapter branch."""

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
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.volterra_scaling = alpha / rank
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.A_v = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.B_v = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_v.weight)

    def freeze_base(self) -> None:
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.A_v(x)
        return self.conv(x) + self.B_v(z * z) * self.volterra_scaling


class LinearVolterraConv2d(nn.Module):
    """LoRA plus a second low-rank linear 1x1 branch without quadratic interaction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lora_rank: int = 8,
        volterra_rank: int = 8,
        lora_alpha: float = 1.0,
        volterra_alpha: float = 1.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if lora_rank <= 0 or volterra_rank <= 0:
            raise ValueError("adapter ranks must be positive.")
        self.lora_scaling = lora_alpha / lora_rank
        self.volterra_scaling = volterra_alpha / volterra_rank
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.A_l = nn.Conv2d(in_channels, lora_rank, kernel_size=1, bias=False)
        self.B_l = nn.Conv2d(lora_rank, out_channels, kernel_size=1, bias=False)
        self.A_v = nn.Conv2d(in_channels, volterra_rank, kernel_size=1, bias=False)
        self.B_v = nn.Conv2d(volterra_rank, out_channels, kernel_size=1, bias=False)
        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)
        nn.init.kaiming_uniform_(self.A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_v.weight)

    def freeze_base(self) -> None:
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora = self.B_l(self.A_l(x)) * self.lora_scaling
        linear_volterra = self.B_v(self.A_v(x)) * self.volterra_scaling
        return self.conv(x) + lora + linear_volterra
