import torch
from torch import nn


class InputRouter(nn.Module):
    """Softmax router that maps pooled input features to expert weights."""

    def __init__(self, in_features: int, num_experts: int) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")

        self.in_features = in_features
        self.num_experts = num_experts
        self.proj = nn.Linear(in_features, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            pooled = x
        elif x.ndim == 3:
            pooled = x.mean(dim=1)
        else:
            raise ValueError("InputRouter expects shape (batch, features) or (batch, tokens, features).")

        return torch.softmax(self.proj(pooled), dim=-1)
