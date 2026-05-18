"""Parity between full-prefix forward and KV-cache prefill/decode."""

import torch

from stage.conditioning.embedded_condition import EmbeddedCondition
from stage.hyperparameters import LmParams
from stage.models.musicgen_lm import MusicgenLm


def _make_tiny_lm() -> MusicgenLm:
    params = LmParams(
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
        full_logits[..., :prefix_len],
        prefill_logits,
        rtol=1e-4,
        atol=1e-5,
    ), "Prefill logits must match full forward on prefix"

    for step in range(prefix_len, t):
        step_logits = lm.decode_step(x[..., step:step + 1], cache, timestep=step)
        assert torch.allclose(
            full_logits[..., step:step + 1],
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
    assert out.shape[-2] == 8
