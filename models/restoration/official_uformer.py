from __future__ import annotations

import importlib.util
from pathlib import Path

from torch import nn


def _load_uformer_class():
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "external" / "Uformer" / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"Uformer model.py not found at {model_path}")
    spec = importlib.util.spec_from_file_location("official_uformer_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Uformer from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Uformer


def build_official_uformer(size: str = "tiny", img_size: int = 128) -> nn.Module:
    Uformer = _load_uformer_class()
    configs = {
        "tiny": dict(embed_dim=16, depths=[1, 1, 1, 1, 1, 1, 1, 1, 1], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2]),
        "base": dict(embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2]),
    }
    if size not in configs:
        raise ValueError(f"Unknown Uformer size: {size}")
    return Uformer(
        img_size=img_size,
        in_chans=3,
        dd_in=3,
        win_size=8,
        token_projection="linear",
        token_mlp="ffn",
        shift_flag=True,
        modulator=False,
        **configs[size],
    )
