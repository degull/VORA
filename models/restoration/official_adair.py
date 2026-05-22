from __future__ import annotations

import importlib.util
from pathlib import Path

from torch import nn


def _load_adair_class():
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "external" / "AdaIR" / "net" / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"AdaIR model.py not found at {model_path}")
    spec = importlib.util.spec_from_file_location("official_adair_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load AdaIR from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AdaIR


def build_official_adair(size: str = "tiny", img_size: int = 128) -> nn.Module:
    AdaIR = _load_adair_class()
    configs = {
        "tiny": dict(dim=24, num_blocks=[1, 1, 1, 1], num_refinement_blocks=1, heads=[1, 2, 4, 8]),
        "base": dict(dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4, heads=[1, 2, 4, 8]),
    }
    if size not in configs:
        raise ValueError(f"Unknown AdaIR size: {size}")
    return AdaIR(inp_channels=3, out_channels=3, decoder=True, **configs[size])
