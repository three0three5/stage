"""Autoregressive LM generation loop (KV-safe); importable without Lightning/Encodec."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor
from tqdm import tqdm

from stage.conditioning.conditioning_method import ConditioningMethod
from stage.conditioning.embedded_condition import EmbeddedCondition
from stage.inference_config import InferenceConfig
from stage.models.lm_cache import LmInferenceCache
from stage.models.musicgen_lm import MusicgenLm
from stage.utils.sample import sample_top_k


def logits_to_next_token(
    logits: Tensor,
    n_cond_batch: int,
    inference_cfg: InferenceConfig,
) -> Tensor:
    cond_logits, uncond_logits = logits.split(n_cond_batch, dim=0)
    logits = uncond_logits + (cond_logits - uncond_logits) * inference_cfg.cfg_coef
    logits = logits.permute(0, 1, 3, 2)[..., -1]
    if inference_cfg.greedy:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1)
    return sample_top_k(probs, k=inference_cfg.top_k)


def lm_logits(
    lm: MusicgenLm,
    gen_sequence: Tensor,
    attention_mask: Tensor,
    method_to_cond: Dict[ConditioningMethod, EmbeddedCondition],
    end_offset: int,
) -> Tensor:
    return lm(
        x=gen_sequence[..., :end_offset],
        attention_mask=attention_mask[..., :end_offset],
        cross_attention_input=method_to_cond.get(
            ConditioningMethod.CROSS_ATTENTION),
        prepend_embeds=method_to_cond.get(ConditioningMethod.INPUT_PREPEND),
        sum_embeds=method_to_cond.get(ConditioningMethod.INPUT_SUM),
    )


def prefill_kv(
    lm: MusicgenLm,
    gen_sequence: Tensor,
    attention_mask: Tensor,
    method_to_cond: Dict[ConditioningMethod, EmbeddedCondition],
    end_offset: int,
) -> tuple[Tensor, LmInferenceCache]:
    return lm.prefill(
        gen_sequence[..., :end_offset],
        attention_mask[..., :end_offset],
        cross_attention_input=method_to_cond.get(
            ConditioningMethod.CROSS_ATTENTION),
        prepend_embeds=method_to_cond.get(ConditioningMethod.INPUT_PREPEND),
        sum_embeds=method_to_cond.get(ConditioningMethod.INPUT_SUM),
    )


def sample_next_token_kv(
    lm: MusicgenLm,
    gen_sequence: Tensor,
    attention_mask: Tensor,
    method_to_cond: Dict[ConditioningMethod, EmbeddedCondition],
    n_samples: int,
    offset: int,
    kv_cache: LmInferenceCache,
    inference_cfg: InferenceConfig,
) -> tuple[Tensor, LmInferenceCache, bool]:
    """Incremental decode; optional ref-forward parity check when verify_kv_parity."""
    inc_logits = lm.decode_step(
        gen_sequence[..., offset - 1:offset],
        kv_cache,
        offset - 1,
        attention_mask,
    )
    if not inference_cfg.verify_kv_parity:
        return (
            logits_to_next_token(inc_logits, n_samples, inference_cfg),
            kv_cache,
            False,
        )

    ref_logits = lm_logits(lm, gen_sequence, attention_mask, method_to_cond,
                           offset)
    ref_step = ref_logits[..., offset - 1:offset, :]
    if torch.allclose(ref_step, inc_logits, rtol=1e-4, atol=1e-5):
        return (
            logits_to_next_token(inc_logits, n_samples, inference_cfg),
            kv_cache,
            False,
        )
    _, kv_cache = prefill_kv(lm, gen_sequence, attention_mask, method_to_cond,
                             offset)
    return (
        logits_to_next_token(ref_step, n_samples, inference_cfg),
        kv_cache,
        True,
    )


def write_token_at_offset(
    gen_sequence: Tensor,
    gen_mask: Tensor,
    n_samples: int,
    offset: int,
    next_token: Tensor,
    special_token: int,
) -> None:
    valid_mask = gen_mask[..., offset:offset + 1].expand(n_samples, -1, -1)
    next_token = next_token.clone()
    next_token[~valid_mask] = special_token
    for batch_slice in (slice(n_samples), slice(n_samples, None)):
        gen_sequence[batch_slice, :, offset:offset + 1] = torch.where(
            gen_sequence[batch_slice, :, offset:offset + 1] == -1,
            next_token,
            gen_sequence[batch_slice, :, offset:offset + 1],
        )


def autoregressive_lm_loop(
    lm: MusicgenLm,
    special_token: int,
    gen_sequence: Tensor,
    gen_mask: Tensor,
    attention_mask: Tensor,
    start_offset: int,
    method_to_cond: Dict[ConditioningMethod, EmbeddedCondition],
    n_samples: int,
    inference_cfg: InferenceConfig,
    prog_bar: bool = False,
) -> Tensor:
    """Fill interleaved LM sequence (CFG batch); returns stacked gen_sequence."""
    kv_cache: Optional[LmInferenceCache] = None
    use_kv = inference_cfg.use_kv_cache and start_offset > 0

    if use_kv:
        logits, kv_cache = prefill_kv(lm, gen_sequence, attention_mask,
                                      method_to_cond, start_offset)
        next_token = logits_to_next_token(logits, n_samples, inference_cfg)
        write_token_at_offset(gen_sequence, gen_mask, n_samples, start_offset,
                              next_token, special_token)

    iterator = range(
        start_offset + 1 if use_kv else start_offset,
        gen_sequence.shape[-1],
    )
    if prog_bar:
        iterator = tqdm(iterator, desc="generating autoregressively...")
    for offset in iterator:
        if use_kv and kv_cache is not None:
            next_token, kv_cache, _resynced = sample_next_token_kv(
                lm,
                gen_sequence,
                attention_mask,
                method_to_cond,
                n_samples,
                offset,
                kv_cache,
                inference_cfg,
            )
        else:
            logits = lm_logits(lm, gen_sequence, attention_mask, method_to_cond,
                               offset)
            next_token = logits_to_next_token(logits, n_samples, inference_cfg)
        write_token_at_offset(gen_sequence, gen_mask, n_samples, offset,
                              next_token, special_token)

    assert not (gen_sequence == -1).any()
    return gen_sequence
