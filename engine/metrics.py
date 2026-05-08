import math

import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).item()
    return 20.0 * math.log10(1.0 / math.sqrt(max(mse, eps)))


def simple_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = ((pred - mu_x) ** 2).mean()
    sigma_y = ((target - mu_y) ** 2).mean()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.item())
