from __future__ import annotations

import torch
from torch import nn


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int, batch: int) -> torch.Tensor:
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class SwinIRLiteBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, window_size: int = 8) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        shortcut = x
        b, _, c = x.shape
        x_img = self.norm1(x).view(b, height, width, c)
        windows = window_partition(x_img, self.window_size)
        windows = self.attn(windows)
        x = window_reverse(windows, self.window_size, height, width, b).view(b, height * width, c)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class SwinIRLite(nn.Module):
    """Small Linear-projection restoration transformer used to wire up VoRA training."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 48,
        depth: int = 4,
        num_heads: int = 4,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shallow = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                SwinIRLiteBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.reconstruct = nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        feat = self.shallow(x)
        b, c, height, width = feat.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h or pad_w:
            feat = nn.functional.pad(feat, (0, pad_w, 0, pad_h), mode="reflect")

        _, _, padded_h, padded_w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens, padded_h, padded_w)
        feat = self.norm(tokens).transpose(1, 2).view(b, c, padded_h, padded_w)
        out = self.reconstruct(feat)
        if pad_h or pad_w:
            out = out[:, :, :height, :width]
        return torch.clamp(residual + out, 0.0, 1.0)
