import math

import torch
from torch import nn


class QuadraticVolterra(nn.Module):
    """Efficient second-order Volterra branch in a low-rank feature space."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")

        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.A_v = nn.Linear(in_features, rank, bias=False)
        self.B_v = nn.Linear(rank, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_v.weight)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        z = self.A_v(self.dropout(x))
        if context is None:
            context = z
        return self.B_v(z * context) * self.scaling
