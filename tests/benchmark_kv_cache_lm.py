"""Lightweight LM-only benchmark: full forward vs KV prefill/decode (no checkpoint)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import time

import tests.conftest  # noqa: F401 — lightweight import bootstrap

import torch

from stage.conditioning.embedded_condition import EmbeddedCondition
from stage.models.musicgen_lm import MusicgenLm
from tests.test_kv_cache_parity import _TinyLmParams, _make_tiny_lm, _random_batch


def _bench(fn, warmup: int = 2, reps: int = 3) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = _make_tiny_lm().to(device).eval()

    b, k, t = 1, 4, 64
    prefix_len = 32
    n_decode = t - prefix_len

    x = _random_batch(b, k, t, lm.card).to(device)
    mask = torch.ones(b, t, dtype=torch.bool, device=device)
    cross = EmbeddedCondition(
        data=torch.randn(b, 8, lm.dim, device=device),
        mask=torch.ones(b, 8, dtype=torch.bool, device=device),
    )

    def full_autoregressive() -> None:
        for step in range(prefix_len, t):
            lm(x[..., :step + 1], mask[:, :step + 1], cross_attention_input=cross)

    def kv_autoregressive() -> None:
        _, cache = lm.prefill(
            x[..., :prefix_len],
            mask[:, :prefix_len],
            cross_attention_input=cross,
        )
        for step in range(prefix_len, t):
            lm.decode_step(
                x[..., step:step + 1],
                cache,
                timestep=step,
                attention_mask=mask,
            )

    t_full = _bench(full_autoregressive)
    t_kv = _bench(kv_autoregressive)

    print(f"device={device}")
    print(f"decode_steps={n_decode} (prefix={prefix_len}, total={t})")
    print(f"full_forward_per_step_s={t_full / n_decode:.4f}")
    print(f"kv_cache_per_step_s={t_kv / n_decode:.4f}")
    print(f"speedup={t_full / t_kv:.2f}x")


if __name__ == "__main__":
    main()
