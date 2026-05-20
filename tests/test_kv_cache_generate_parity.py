"""KV vs full-forward parity under interleaved-style masks and generate-loop tokens."""

from __future__ import annotations

import pytest
import torch

from stage.conditioning.conditioning_method import ConditioningMethod
from stage.conditioning.embedded_condition import EmbeddedCondition
from stage.inference_config import InferenceConfig
from stage.models import lm_autoregressive as ar

from test_kv_cache_parity import _make_tiny_lm


def _delayed_style_masks(
    b: int,
    k: int,
    t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate delayed interleaving: early steps only higher-index codebooks active."""
    gen_mask = torch.ones(b, k, t, dtype=torch.bool)
    for s in range(t):
        for qi in range(k):
            if s < qi + 1:
                gen_mask[:, qi, s] = False
    attention_mask = gen_mask.any(dim=1)
    return gen_mask, attention_mask


def _synthetic_generate_batch(
    b_cfg: int,
    k: int,
    t: int,
    start_offset: int,
    card: int,
    special_token: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CFG-doubled gen_sequence, cond gen_mask, attention_mask."""
    torch.manual_seed(0)
    gen_sequence = torch.full((b_cfg, k, t), -1, dtype=torch.long)
    prefix = torch.randint(0, card, (b_cfg, k, start_offset))
    gen_sequence[..., :start_offset] = prefix
    n_samples = b_cfg // 2
    gen_mask, _ = _delayed_style_masks(n_samples, k, t)
    # Mark invalid conditioning specials (cond batch only; avoid broadcasting gen_mask).
    cond = torch.zeros_like(gen_mask)
    cond[:, :, : max(0, start_offset - 3)] = True
    invalid = (gen_sequence[:n_samples] == special_token) & cond
    gen_mask = (~invalid) & gen_mask
    attention_mask_cfg = gen_mask.any(dim=1).repeat(2, 1)
    return gen_sequence, gen_mask, attention_mask_cfg


@torch.no_grad()
def test_lm_logits_parity_delayed_mask() -> None:
    torch.manual_seed(1)
    lm = _make_tiny_lm()
    b, k, t = 2, 4, 20
    card = lm.card

    x = torch.randint(0, card, (b, k, t))
    gen_mask, attention_mask = _delayed_style_masks(b, k, t)
    x = torch.where(gen_mask, x, torch.full_like(x, 2048))

    cross = EmbeddedCondition(
        data=torch.randn(b, 6, lm.dim),
        mask=torch.ones(b, 6, dtype=torch.bool),
    )

    full_logits = lm(x, attention_mask, cross_attention_input=cross)

    prefix_len = 12
    prefix = x[..., :prefix_len]
    prefix_mask = attention_mask[:, :prefix_len]

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
    )

    for step in range(prefix_len, t):
        step_logits = lm.decode_step(
            x[..., step:step + 1],
            cache,
            timestep=step,
            attention_mask=attention_mask,
        )
        ref = full_logits[:, :, step:step + 1, :]
        assert torch.allclose(
            ref,
            step_logits,
            rtol=1e-4,
            atol=1e-5,
        ), f"Decode step {step} diverged with delayed-style mask"


@torch.no_grad()
def test_autoregressive_loop_tokens_match_kv() -> None:
    lm = _make_tiny_lm().eval()
    special_token = 2048
    n_samples = 1
    b_cfg = 2 * n_samples
    k, t = 4, 24
    start_offset = 10
    card = lm.card
    gen_sequence, gen_mask, attention_mask = _synthetic_generate_batch(
        b_cfg, k, t, start_offset, card, special_token)

    cross = EmbeddedCondition(
        data=torch.randn(b_cfg, 6, lm.dim),
        mask=torch.ones(b_cfg, 6, dtype=torch.bool),
    )
    method_to_cond = {ConditioningMethod.CROSS_ATTENTION: cross}

    def run(use_kv: bool) -> torch.Tensor:
        torch.manual_seed(99)
        seq = gen_sequence.clone()
        cfg = InferenceConfig(use_kv_cache=use_kv, greedy=True)
        return ar.autoregressive_lm_loop(
            lm,
            special_token,
            seq,
            gen_mask,
            attention_mask,
            start_offset,
            method_to_cond,
            n_samples,
            cfg,
        )

    out_no_kv = run(use_kv=False)
    out_kv = run(use_kv=True)

    assert torch.equal(out_no_kv[:n_samples], out_kv[:n_samples]), (
        "Cond batch tokens must match between KV and full-forward loops"
    )


@torch.no_grad()
@pytest.mark.slow
def test_autoregressive_loop_tokens_match_kv_real_weights() -> None:
    from pathlib import Path

    ckp = Path("checkpoints/stage-bass.safetensors")
    if not ckp.is_file():
        pytest.skip("stage-bass checkpoint not available")

    from stage.models.lightning_musicgen import LightningMusicgen

    model = LightningMusicgen.load_from_checkpoint_replacing_paths(ckp)
    model.eval()
    model.to("cpu")

    n_samples = 1
    b_cfg = 2
    k = model.n_q
    t = 32
    start_offset = 14
    card = model.lm.card

    gen_sequence, gen_mask, attention_mask = _synthetic_generate_batch(
        b_cfg, k, t, start_offset, card, model.special_token)

    method_to_cond: dict = {}

    def run(use_kv: bool) -> torch.Tensor:
        torch.manual_seed(7)
        seq = gen_sequence.clone()
        cfg = InferenceConfig(
            use_kv_cache=use_kv,
            greedy=True,
            use_fp16=False,
        )
        return model._autoregressive_lm_loop(
            seq,
            gen_mask,
            attention_mask,
            start_offset,
            method_to_cond,
            n_samples,
            cfg,
        )

    assert torch.equal(run(False)[:1], run(True)[:1])
