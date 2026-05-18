"""Benchmark STAGE inference latency (wall time, steps/s, RTF)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from stage import config as cfg
from stage.inference_config import InferenceConfig
from stage.loader import load_for_inference
from stage.utils.audio import load_audio
from stage.utils.device import sync_device


def run_benchmark(
    checkpoint: Path,
    sample_wav: Path | None,
    gen_seconds: float,
    inference_cfg: InferenceConfig,
    warmup: int = 1,
) -> dict:
    device = inference_cfg.resolved_device()
    model = load_for_inference(checkpoint, inference_cfg=inference_cfg)

    context = None
    desc = ["upbeat rock drums"]
    if sample_wav is not None and sample_wav.exists():
        context = load_audio(sample_wav).to(device)
        desc_path = sample_wav.parent / f"{sample_wav.stem}-desc.txt"
        if desc_path.exists():
            desc = [desc_path.read_text().strip()]

    for _ in range(warmup):
        model.generate(
            n_samples=1,
            gen_seconds=min(gen_seconds, 2.0),
            prompt=None,
            context=context,
            style=None,
            beat=None,
            description=desc,
            inference_cfg=inference_cfg,
        )
    sync_device(device)

    start = time.perf_counter()
    out = model.generate(
        n_samples=1,
        gen_seconds=gen_seconds,
        prompt=None,
        context=context,
        style=None,
        beat=None,
        description=desc,
        inference_cfg=inference_cfg,
    )
    sync_device(device)
    elapsed = time.perf_counter() - start

    n_steps = int(model.encodec_model.frame_rate * gen_seconds)
    audio_sec = out.shape[-1] / model.sample_rate
    return {
        "elapsed_s": elapsed,
        "gen_seconds": gen_seconds,
        "lm_steps": n_steps,
        "steps_per_s": n_steps / elapsed if elapsed > 0 else 0.0,
        "rtf": elapsed / audio_sec if audio_sec > 0 else 0.0,
        "device": str(device),
        "kv_cache": inference_cfg.use_kv_cache,
        "fp16": inference_cfg.use_fp16,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark STAGE inference")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=cfg.CKP_DIR / "stage-drums.safetensors",
    )
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--gen-seconds", type=float, default=10.0)
    parser.add_argument("--no-kv-cache", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    inference_cfg = InferenceConfig(
        use_kv_cache=not args.no_kv_cache,
        device=args.device,
    )
    stats = run_benchmark(
        args.checkpoint,
        args.audio,
        args.gen_seconds,
        inference_cfg,
    )
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
