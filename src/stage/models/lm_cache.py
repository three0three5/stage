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

    # Key padding mask for cached self-attention (prepend + sequence timesteps)
    key_valid_mask: Optional[Tensor] = None

    def clone_for_cfg(self) -> "LmInferenceCache":
        """Shallow copy; cache tensors are shared (CFG cond/uncond in one batch)."""
        return LmInferenceCache(
            layer_intermediates=self.layer_intermediates,
            prepend_len=self.prepend_len,
            first_valid_indices=self.first_valid_indices,
            cross_attention_data=self.cross_attention_data,
            cross_attention_mask=self.cross_attention_mask,
            sum_embeds_data=self.sum_embeds_data,
            key_valid_mask=self.key_valid_mask,
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


def cached_kv_seq_len(layer_intermediates: Any) -> Optional[int]:
    """Length of cached K along the sequence axis (first layer with ``cached_kv``)."""
    attn = getattr(layer_intermediates, "attn_intermediates", None)
    if attn is None:
        return None
    for inter in attn:
        cached_kv = getattr(inter, "cached_kv", None)
        if cached_kv is not None:
            return cached_kv[0].shape[-2]
    return None


def align_key_mask_to_cached_kv(
    key_mask: Tensor,
    layer_intermediates: Any,
    query_len: int,
) -> Tensor:
    """Match ``key_mask`` length to cached K/V plus current query tokens (flash-attn)."""
    kv_len = cached_kv_seq_len(layer_intermediates)
    if kv_len is None:
        return key_mask
    target_len = kv_len + query_len
    mask_len = key_mask.shape[-1]
    if mask_len == target_len:
        return key_mask
    if mask_len > target_len:
        return key_mask[:, -target_len:]
    raise RuntimeError(
        f"key_valid_mask length {mask_len} is shorter than KV+query length "
        f"{target_len} (kv_len={kv_len}, query_len={query_len})")


def merge_kv_cache_intermediates(
    old: Any,
    new: Any,
) -> Any:
    """Merge KV only when ``new`` holds a single-step delta (older x-transformers builds).

    x-transformers >= 1.26 already stores cumulative K/V in ``cached_kv``; in that case
    return ``new`` unchanged to avoid double-counting history.
    """
    old_attn = getattr(old, "attn_intermediates", None)
    new_attn = getattr(new, "attn_intermediates", None)
    if old_attn is None or new_attn is None:
        return new
    for old_inter, new_inter in zip(old_attn, new_attn):
        new_kv = getattr(new_inter, "cached_kv", None)
        if new_kv is None:
            continue
        old_kv = getattr(old_inter, "cached_kv", None)
        if old_kv is None:
            new_inter.cached_kv = new_kv
            continue
        nk_len = new_kv[0].shape[-2]
        ok_len = old_kv[0].shape[-2]
        if nk_len <= 1 and ok_len >= 1:
            ok, ov = old_kv
            nk, nv = new_kv
            new_inter.cached_kv = (
                torch.cat([ok, nk], dim=-2),
                torch.cat([ov, nv], dim=-2),
            )
    return new
