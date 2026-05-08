from torch import nn


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
