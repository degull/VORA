from torch import nn


def build_l1_loss() -> nn.Module:
    return nn.L1Loss()
