"""Device helpers for cross-platform inference."""

from typing import Optional
import torch


def sync_device(device: torch.device) -> None:
    """Block until pending ops finish (for accurate benchmarking only)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
