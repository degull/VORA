import math

import torch
from torch import nn

from .routing import InputRouter
from .token_interaction import LocalTokenInteraction


class VoRATokenLinear(nn.Module):
    """VoRA layer with local token interaction context."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int = 8,
        volterra_rank: int = 8,
        lora_alpha: float = 1.0,
        volterra_alpha: float = 1.0,
        token_window_size: int = 3,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.lora_scaling = lora_alpha / lora_rank
        self.volterra_scaling = volterra_alpha / volterra_rank
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.A_l = nn.Linear(in_features, lora_rank, bias=False)
        self.B_l = nn.Linear(lora_rank, out_features, bias=False)
        self.A_v = nn.Linear(in_features, volterra_rank, bias=False)
        self.token_interaction = LocalTokenInteraction(volterra_rank, window_size=token_window_size)
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
        lora = self.B_l(self.A_l(x)) * self.lora_scaling
        z = self.A_v(x)
        context = self.token_interaction(z) if z.ndim == 3 else z
        volterra = self.B_v(z * context) * self.volterra_scaling
        return base + lora + volterra


class VoRAFullLinear(nn.Module):
    """VoRA-token with multiple input-routed Volterra experts."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int = 8,
        volterra_rank: int = 8,
        num_experts: int = 4,
        lora_alpha: float = 1.0,
        volterra_alpha: float = 1.0,
        token_window_size: int = 3,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.lora_scaling = lora_alpha / lora_rank
        self.volterra_scaling = volterra_alpha / volterra_rank
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.A_l = nn.Linear(in_features, lora_rank, bias=False)
        self.B_l = nn.Linear(lora_rank, out_features, bias=False)
        self.router = InputRouter(in_features, num_experts)
        self.A_v = nn.ModuleList([nn.Linear(in_features, volterra_rank, bias=False) for _ in range(num_experts)])
        self.interactions = nn.ModuleList(
            [LocalTokenInteraction(volterra_rank, window_size=token_window_size) for _ in range(num_experts)]
        )
        self.B_v = nn.ModuleList([nn.Linear(volterra_rank, out_features, bias=False) for _ in range(num_experts)])
        self.reset_parameters()
        if freeze_base:
            self.freeze_base()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_l.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_l.weight)
        for a_v, b_v in zip(self.A_v, self.B_v):
            nn.init.kaiming_uniform_(a_v.weight, a=math.sqrt(5))
            nn.init.zeros_(b_v.weight)

    def freeze_base(self) -> None:
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = self.B_l(self.A_l(x)) * self.lora_scaling
        alpha = self.router(x)
        expert_outputs = []
        for a_v, interaction, b_v in zip(self.A_v, self.interactions, self.B_v):
            z = a_v(x)
            context = interaction(z) if z.ndim == 3 else z
            expert_outputs.append(b_v(z * context) * self.volterra_scaling)

        stacked = torch.stack(expert_outputs, dim=1)
        while alpha.ndim < stacked.ndim:
            alpha = alpha.unsqueeze(-1)
        routed = (stacked * alpha).sum(dim=1)
        return base + lora + routed
