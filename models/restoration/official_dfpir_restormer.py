from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from torch import nn


def _install_optional_stubs() -> None:
    if "clip" not in sys.modules:
        sys.modules["clip"] = types.ModuleType("clip")


def _load_restormer_class():
    repo_root = Path(__file__).resolve().parents[2]
    dfpir_root = repo_root / "external" / "DFPIR"
    model_path = dfpir_root / "net" / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(
            "DFPIR model.py not found. Clone the official DFPIR repository to "
            f"{dfpir_root} before running the dfpir_restormer backbone."
        )
    _install_optional_stubs()
    if str(dfpir_root) not in sys.path:
        sys.path.insert(0, str(dfpir_root))
    spec = importlib.util.spec_from_file_location("official_dfpir_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DFPIR model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Restormer


def build_official_dfpir_restormer(size: str = "base", img_size: int = 128) -> nn.Module:
    del img_size
    Restormer = _load_restormer_class()
    configs = {
        "tiny": dict(dim=24, num_blocks=[1, 1, 1, 1], num_refinement_blocks=1, heads=[1, 2, 4, 8]),
        "base": dict(dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4, heads=[1, 2, 4, 8]),
    }
    if size not in configs:
        raise ValueError(f"Unknown DFPIR/Restormer size: {size}")
    return Restormer(inp_channels=3, out_channels=3, **configs[size])
