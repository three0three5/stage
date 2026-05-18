"""KV-cache state for MusicgenLm incremental decoding (x-transformers LayerIntermediates)."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class LmInferenceCache:
    """Holds x-transformers attention cache and position bookkeeping."""

    # x_transformers.x_transformers.LayerIntermediates
    layer_intermediates: Any
    prepend_len: int = 0
    first_valid_indices: Optional[Tensor] = None

    # Frozen conditioning passed on every decode step
    cross_attention_data: Optional[Tensor] = None
    cross_attention_mask: Optional[Tensor] = None
    sum_embeds_data: Optional[Tensor] = None

    def clone_for_cfg(self) -> "LmInferenceCache":
        """Shallow copy; cache tensors are shared (CFG cond/uncond in one batch)."""
        return LmInferenceCache(
            layer_intermediates=self.layer_intermediates,
            prepend_len=self.prepend_len,
            first_valid_indices=self.first_valid_indices,
            cross_attention_data=self.cross_attention_data,
            cross_attention_mask=self.cross_attention_mask,
            sum_embeds_data=self.sum_embeds_data,
        )


def position_for_timestep(
    timestep: int,
    first_valid_indices: Tensor,
    prepend_len: int,
    device: torch.device,
) -> Tensor:
    """Absolute sinusoidal position for a single sequence index (B, 1, 1)."""
    rel = timestep - first_valid_indices
    pos = (rel + prepend_len).clamp(min=0)
    return pos.view(-1, 1, 1).to(device=device)
