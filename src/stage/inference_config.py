"""Runtime configuration for STAGE inference (separate from training hyperparameters)."""

from dataclasses import dataclass
from typing import Optional
import torch


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Pick the best available device: explicit > cuda > mps > cpu."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(device: torch.device, use_fp16: bool) -> torch.dtype:
    if not use_fp16:
        return torch.float32
    if device.type in ("cuda", "mps"):
        return torch.float16
    return torch.float32


@dataclass
class InferenceConfig:
    """Knobs for local / production inference."""

    cfg_coef: float = 3.0
    top_k: int = 250
    use_kv_cache: bool = True
    use_fp16: bool = True
    compile_decode: bool = False
    t5_on_cpu: bool = True
    device: Optional[str] = None
    attn_flash: Optional[bool] = None  # None = auto (off on mps)

    def resolved_device(self) -> torch.device:
        return resolve_device(self.device)

    def resolved_attn_flash(self, device: torch.device) -> bool:
        if self.attn_flash is not None:
            return self.attn_flash
        return device.type != "mps"
