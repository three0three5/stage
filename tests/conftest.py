"""Pytest setup: lightweight imports for KV-cache unit tests (no full STAGE stack)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_submodule(package: str, module_name: str, rel_path: str) -> types.ModuleType:
    full_name = f"{package}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    path = SRC / rel_path
    spec = importlib.util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {full_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub stage.conditioning so ``embedded_condition`` does not pull beat_embedder/audio/stem.
if "stage" not in sys.modules:
    sys.modules["stage"] = types.ModuleType("stage")
    sys.modules["stage"].__path__ = [str(SRC / "stage")]

cond_pkg = types.ModuleType("stage.conditioning")
cond_pkg.__path__ = [str(SRC / "stage" / "conditioning")]
sys.modules["stage.conditioning"] = cond_pkg

_load_submodule("stage.conditioning", "embedded_condition",
               "stage/conditioning/embedded_condition.py")
_load_submodule("stage.conditioning", "condition_type",
               "stage/conditioning/condition_type.py")
_load_submodule("stage.conditioning", "conditioning_method",
               "stage/conditioning/conditioning_method.py")

# Stub config (real module uses 3.10+ syntax; tests only need the import to exist).
config_mod = types.ModuleType("stage.config")
config_mod.weights_dir = lambda: ROOT / "weights"
config_mod.CKP_DIR = ROOT / "checkpoints"
sys.modules["stage.config"] = config_mod

# Stub hyperparameters so musicgen_lm can import without the full training stack.
from dataclasses import dataclass
from typing import Optional as _Optional


@dataclass
class _LmParams:
    dim: int
    n_layers: int
    n_heads: int
    card: int = 2048
    padding_token: _Optional[int] = 2048
    sep_token: _Optional[int] = None
    cross_attend: bool = True
    weights: _Optional[str] = None


hp_mod = types.ModuleType("stage.hyperparameters")
hp_mod.LmParams = _LmParams
sys.modules["stage.hyperparameters"] = hp_mod
