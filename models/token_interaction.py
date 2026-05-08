import torch
from torch import nn


class LocalTokenInteraction(nn.Module):
    """A simple local token interaction module for sequence-shaped features."""

    def __init__(self, rank: int, window_size: int = 3) -> None:
        super().__init__()
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer.")

        self.rank = rank
        self.window_size = window_size
        self.depthwise = nn.Conv1d(
            in_channels=rank,
            out_channels=rank,
            kernel_size=window_size,
            padding=window_size // 2,
            groups=rank,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.depthwise.weight, 1.0 / self.window_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError("LocalTokenInteraction expects shape (batch, tokens, rank).")

        context = self.depthwise(z.transpose(1, 2)).transpose(1, 2)
        return context
