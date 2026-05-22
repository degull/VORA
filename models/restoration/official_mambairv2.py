from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from torch import nn


def _load_mambairv2_class():
    repo_root = Path(__file__).resolve().parents[2]
    mambair_root = repo_root / "external" / "MambaIR"
    arch_path = mambair_root / "basicsr" / "archs" / "mambairv2_arch.py"
    if not arch_path.exists():
        raise FileNotFoundError(f"MambaIRv2 arch not found at {arch_path}")
    if str(mambair_root) not in sys.path:
        sys.path.insert(0, str(mambair_root))
    spec = importlib.util.spec_from_file_location("official_mambairv2_arch", arch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MambaIRv2 from {arch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MambaIRv2


def build_official_mambairv2(size: str = "tiny", img_size: int = 128) -> nn.Module:
    MambaIRv2 = _load_mambairv2_class()
    configs = {
        "tiny": dict(embed_dim=48, depths=(2, 2), num_heads=(4, 4), window_size=8, inner_rank=16, num_tokens=32),
        "base": dict(embed_dim=48, depths=(6, 6, 6, 6), num_heads=(4, 4, 4, 4), window_size=16),
    }
    if size not in configs:
        raise ValueError(f"Unknown MambaIRv2 size: {size}")
    return MambaIRv2(
        img_size=img_size,
        patch_size=1,
        in_chans=3,
        upscale=1,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        **configs[size],
    )
