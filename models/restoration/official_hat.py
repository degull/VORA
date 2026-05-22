from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from torch import nn
import torch
from timm.models.layers import to_2tuple, trunc_normal_


class HATRestorationWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        self.model.mean = self.model.mean.type_as(x)
        x = (x - self.model.mean) * self.model.img_range
        x_first = self.model.conv_first(x)
        res = self.model.conv_after_body(self.model.forward_features(x_first)) + x_first
        x = x + self.model.conv_last(res)
        return x / self.model.img_range + self.model.mean


class _DummyRegistry:
    def register(self):
        def decorator(cls):
            return cls

        return decorator


def _install_basicsr_stubs() -> None:
    basicsr = types.ModuleType("basicsr")
    basicsr_utils = types.ModuleType("basicsr.utils")
    basicsr_registry = types.ModuleType("basicsr.utils.registry")
    basicsr_archs = types.ModuleType("basicsr.archs")
    basicsr_arch_util = types.ModuleType("basicsr.archs.arch_util")

    basicsr_registry.ARCH_REGISTRY = _DummyRegistry()
    basicsr_arch_util.to_2tuple = to_2tuple
    basicsr_arch_util.trunc_normal_ = trunc_normal_

    sys.modules["basicsr"] = basicsr
    sys.modules["basicsr.utils"] = basicsr_utils
    sys.modules["basicsr.utils.registry"] = basicsr_registry
    sys.modules["basicsr.archs"] = basicsr_archs
    sys.modules["basicsr.archs.arch_util"] = basicsr_arch_util


def _load_hat_class():
    repo_root = Path(__file__).resolve().parents[2]
    hat_root = repo_root / "external" / "HAT"
    mambair_root = repo_root / "external" / "MambaIR"
    arch_path = hat_root / "hat" / "archs" / "hat_arch.py"
    if not arch_path.exists():
        raise FileNotFoundError(f"HAT arch not found at {arch_path}")
    _install_basicsr_stubs()
    for path in (str(hat_root), str(mambair_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("official_hat_arch", arch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load HAT from {arch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HAT


def build_official_hat(size: str = "tiny", img_size: int = 128) -> nn.Module:
    HAT = _load_hat_class()
    configs = {
        "tiny": dict(embed_dim=48, depths=(2, 2), num_heads=(3, 3), window_size=8, mlp_ratio=2.0),
        "base": dict(embed_dim=96, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6), window_size=8, mlp_ratio=4.0),
    }
    if size not in configs:
        raise ValueError(f"Unknown HAT size: {size}")
    model = HAT(
        img_size=img_size,
        patch_size=1,
        in_chans=3,
        upscale=1,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        **configs[size],
    )
    if not hasattr(model, "conv_last"):
        model.conv_last = nn.Conv2d(model.embed_dim, 3, 3, 1, 1)
    return HATRestorationWrapper(model)
