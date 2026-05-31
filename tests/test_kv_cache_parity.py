"""Parity between full-prefix forward and KV-cache prefill/decode."""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from stage.conditioning.embedded_condition import EmbeddedCondition
from stage.models.lm_cache import cached_kv_seq_len
from stage.models.musicgen_lm import MusicgenLm


@dataclass
class _TinyLmParams:
    """Minimal LM params (avoids importing hyperparameters / data stack)."""
    dim: int
    n_layers: int
    n_heads: int
    card: int = 2048
    padding_token: Optional[int] = 2048
    sep_token: Optional[int] = None
    cross_attend: bool = True
    weights: Optional[str] = None


def _make_tiny_lm() -> MusicgenLm:
    params = _TinyLmParams(
        dim=128,
        n_layers=2,
        n_heads=4,
        card=2048,
        cross_attend=True,
        padding_token=2048,
        sep_token=None,
        weights=None,
    )
    return MusicgenLm(params).eval()


def _random_batch(b: int, k: int, t: int, card: int) -> torch.Tensor:
    return torch.randint(0, card, (b, k, t))


@torch.no_grad()
def test_kv_cache_matches_full_forward() -> None:
    torch.manual_seed(0)
    lm = _make_tiny_lm()
    b, k, t = 2, 4, 16
    card = lm.card

    x = _random_batch(b, k, t, card)
    mask = torch.ones(b, t, dtype=torch.bool)

    cross = EmbeddedCondition(
        data=torch.randn(b, 8, lm.dim),
        mask=torch.ones(b, 8, dtype=torch.bool),
    )

    full_logits = lm(
        x,
        mask,
        cross_attention_input=cross,
    )

    prefix_len = 10
    prefix = x[..., :prefix_len]
    prefix_mask = mask[:, :prefix_len]

    prefill_logits, cache = lm.prefill(
        prefix,
        prefix_mask,
        cross_attention_input=cross,
    )

    assert torch.allclose(
        full_logits[:, :, :prefix_len, :],
        prefill_logits,
        rtol=1e-4,
        atol=1e-5,
    ), "Prefill logits must match full forward on prefix"

    for step in range(prefix_len, t):
        step_logits = lm.decode_step(
            x[..., step:step + 1],
            cache,
            timestep=step,
            attention_mask=mask,
        )
        assert torch.allclose(
            full_logits[:, :, step:step + 1, :],
            step_logits,
            rtol=1e-4,
            atol=1e-5,
        ), f"Decode step {step} diverged from full forward"


@torch.no_grad()
def test_kv_cache_with_prepend_embeddings() -> None:
    """Prefill + incremental decode matches full forward with INPUT_PREPEND."""
    torch.manual_seed(2)
    lm = _make_tiny_lm()
    b, k, t = 2, 4, 12
    card = lm.card

    x = _random_batch(b, k, t, card)
    mask = torch.ones(b, t, dtype=torch.bool)
    prepend_len = 4
    prepend = EmbeddedCondition(
        data=torch.randn(b, prepend_len, lm.dim),
        mask=torch.ones(b, prepend_len, dtype=torch.bool),
    )

    full_logits = lm(x, mask, prepend_embeds=prepend)

    prefix_len = 7
    prefill_logits, cache = lm.prefill(
        x[..., :prefix_len],
        mask[:, :prefix_len],
        prepend_embeds=prepend,
    )
    assert torch.allclose(
        full_logits[:, :, :prefix_len, :],
        prefill_logits,
        rtol=1e-4,
        atol=1e-5,
    )

    for step in range(prefix_len, t):
        step_logits = lm.decode_step(
            x[..., step:step + 1],
            cache,
            timestep=step,
            attention_mask=mask,
        )
        assert torch.allclose(
            full_logits[:, :, step:step + 1, :],
            step_logits,
            rtol=1e-4,
            atol=1e-5,
        ), f"Decode step {step} diverged with prepend embeddings"


@torch.no_grad()
def test_kv_cache_matches_full_forward_delayed_mask() -> None:
    """Prefill + decode with interleaved-style sparse attention_mask."""
    torch.manual_seed(3)
    lm = _make_tiny_lm()
    b, k, t = 2, 4, 20
    card = lm.card

    x = _random_batch(b, k, t, card)
    gen_mask = torch.ones(b, k, t, dtype=torch.bool)
    for s in range(t):
        for qi in range(k):
            if s < qi + 1:
                gen_mask[:, qi, s] = False
    attention_mask = gen_mask.any(dim=1)
    x = torch.where(gen_mask, x, torch.full_like(x, 2048))

    full_logits = lm(x, attention_mask)

    prefix_len = 12
    prefill_logits, cache = lm.prefill(
        x[..., :prefix_len],
        attention_mask[:, :prefix_len],
    )
    assert torch.allclose(
        full_logits[:, :, :prefix_len, :],
        prefill_logits,
        rtol=1e-4,
        atol=1e-5,
    )

    for step in range(prefix_len, t):
        step_logits = lm.decode_step(
            x[..., step:step + 1],
            cache,
            timestep=step,
            attention_mask=attention_mask,
        )
        assert torch.allclose(
            full_logits[:, :, step:step + 1, :],
            step_logits,
            rtol=1e-4,
            atol=1e-5,
        ), f"Delayed-mask decode step {step} diverged"


@torch.no_grad()
def test_kv_cache_long_decode_matches_full_forward() -> None:
    """Long incremental decode: mask/K aligned and logits match full forward (no trim)."""
    torch.manual_seed(4)
    lm = _make_tiny_lm()
    b, k, t = 2, 4, 32
    card = lm.card
    x = _random_batch(b, k, t, card)
    mask = torch.ones(b, t, dtype=torch.bool)
    prepend = EmbeddedCondition(
        data=torch.randn(b, 1, lm.dim),
        mask=torch.zeros(b, 1, dtype=torch.bool),
    )

    full_logits = lm(x, mask, prepend_embeds=prepend)

    prefix_len = 14
    _, cache = lm.prefill(
        x[..., :prefix_len],
        mask[:, :prefix_len],
        prepend_embeds=prepend,
    )
    assert cache.key_valid_mask is not None

    for step in range(prefix_len, t):
        step_logits = lm.decode_step(
            x[..., step:step + 1],
            cache,
            timestep=step,
            attention_mask=mask,
        )
        kv_len = cached_kv_seq_len(cache.layer_intermediates)
        assert kv_len is not None
        assert cache.key_valid_mask is not None
        assert cache.key_valid_mask.shape[-1] == kv_len, (
            f"step {step}: mask {cache.key_valid_mask.shape[-1]} != kv {kv_len}"
        )
        assert torch.allclose(
            full_logits[:, :, step:step + 1, :],
            step_logits,
            rtol=1e-4,
            atol=1e-5,
        ), f"Decode step {step} diverged from full forward"


@torch.no_grad()
def test_kv_cache_disabled_path_unchanged() -> None:
    """Single forward still works without cache (smoke)."""
    lm = _make_tiny_lm()
    x = _random_batch(1, 4, 8, lm.card)
    mask = torch.ones(1, 8, dtype=torch.bool)
    out = lm(x, mask)
    assert out.shape[2] == 8
