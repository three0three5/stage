"""Benchmark STAGE inference latency (wall time, steps/s, RTF)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

import torch

from stage import config as cfg
from stage.inference_config import InferenceConfig
from stage.loader import load_for_inference
from stage.models.lightning_musicgen import LightningMusicgen
from stage.utils.audio import load_audio, save_audio
from stage.utils.device import sync_device


def _load_context(
    sample_wav: Path | None,
    device: torch.device,
    context_seconds: Optional[float],
    sample_rate: int = 32_000,
) -> Optional[torch.Tensor]:
    if sample_wav is None or not sample_wav.exists():
        return None
    context = load_audio(sample_wav).to(device)
    if context_seconds is not None:
        n_frames = int(sample_rate * context_seconds)
        context = context[..., :n_frames]
    return context


def _resolve_description(
    sample_wav: Path | None,
    null_description: bool,
) -> Optional[List[Optional[str]]]:
    if null_description:
        return [None]
    desc: List[Optional[str]] = ["upbeat rock drums"]
    if sample_wav is not None and sample_wav.exists():
        desc_path = sample_wav.parent / f"{sample_wav.stem}-desc.txt"
        if desc_path.exists():
            desc = [desc_path.read_text().strip()]
    return desc


def run_benchmark(
    model: LightningMusicgen,
    context: Optional[torch.Tensor],
    description: Optional[List[Optional[str]]],
    gen_seconds: float,
    inference_cfg: InferenceConfig,
    warmup: int = 1,
    output_path: Optional[Path] = None,
) -> dict:
    device = inference_cfg.resolved_device()

    for _ in range(warmup):
        model.generate(
            n_samples=1,
            gen_seconds=min(gen_seconds, 2.0),
            prompt=None,
            context=context,
            style=None,
            beat=None,
            description=description,
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
        description=description,
        inference_cfg=inference_cfg,
    )
    sync_device(device)
    elapsed = time.perf_counter() - start

    if output_path is not None:
        save_audio(out, output_path, sample_rate=model.sample_rate)

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
        "output_path": str(output_path.resolve()) if output_path else None,
    }


def _build_inference_cfg(args: argparse.Namespace) -> InferenceConfig:
    attn_flash = None
    if args.attn_flash:
        attn_flash = True
    elif args.no_attn_flash:
        attn_flash = False
    return InferenceConfig(
        use_kv_cache=not args.no_kv_cache,
        use_fp16=not args.no_fp16,
        t5_on_cpu=not args.t5_on_gpu,
        device=args.device,
        attn_flash=attn_flash,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark STAGE inference")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=cfg.CKP_DIR / "stage-bass.safetensors",
    )
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--gen-seconds", type=float, default=10.0)
    parser.add_argument("--context-seconds", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="Save generated audio to this WAV path")
    parser.add_argument("--compare-kv", action="store_true",
                        help="Run twice: use_kv_cache=False then True")
    parser.add_argument("--null-description", action="store_true",
                        help="Use description=[None] (Kaggle-style)")
    parser.add_argument("--no-kv-cache", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-attn-flash", action="store_true")
    parser.add_argument("--attn-flash", action="store_true")
    parser.add_argument("--t5-on-gpu", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    device = InferenceConfig(device=args.device).resolved_device()
    context = _load_context(args.audio, device, args.context_seconds)
    description = _resolve_description(args.audio, args.null_description)

    if args.compare_kv:
        output_dir = args.output_dir or (cfg.ROOT / "outputs" / "local_inference")
        output_dir.mkdir(parents=True, exist_ok=True)
        gen_tag = f"{args.gen_seconds:g}s".replace(".", "p")

        print("Loading model...")
        base_cfg = _build_inference_cfg(args)
        model = load_for_inference(args.checkpoint, inference_cfg=base_cfg)

        results = []
        for use_kv, label in ((False, "no_kv"), (True, "kv")):
            kv_cfg = InferenceConfig(
                use_kv_cache=use_kv,
                use_fp16=base_cfg.use_fp16,
                t5_on_cpu=base_cfg.t5_on_cpu,
                device=base_cfg.device,
                attn_flash=base_cfg.attn_flash,
            )
            out_path = output_dir / f"bass_{label}_{gen_tag}.wav"
            print(f"\n--- use_kv_cache={use_kv} ---")
            stats = run_benchmark(
                model,
                context,
                description,
                args.gen_seconds,
                kv_cfg,
                warmup=args.warmup,
                output_path=out_path,
            )
            results.append(stats)
            for k, v in stats.items():
                print(f"{k}: {v}")

        print("\n=== summary ===")
        for stats in results:
            print(
                f"kv_cache={stats['kv_cache']}: "
                f"elapsed_s={stats['elapsed_s']:.2f}, "
                f"file={stats['output_path']}"
            )
        return

    inference_cfg = _build_inference_cfg(args)
    output_path = args.output
    if output_path is None and args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        gen_tag = f"{args.gen_seconds:g}s".replace(".", "p")
        kv_tag = "kv" if inference_cfg.use_kv_cache else "no_kv"
        output_path = args.output_dir / f"bass_{kv_tag}_{gen_tag}.wav"

    model = load_for_inference(args.checkpoint, inference_cfg=inference_cfg)
    stats = run_benchmark(
        model,
        context,
        description,
        args.gen_seconds,
        inference_cfg,
        warmup=args.warmup,
        output_path=output_path,
    )
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
