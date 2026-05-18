"""Lean inference runtime for local / Mac deployment (no Lightning training path)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor

from stage.conditioning.beat_embedder import Beat
from stage.inference_config import InferenceConfig, resolve_dtype
from stage.loader import load_for_inference
from stage.models.lightning_musicgen import LightningMusicgen
from stage.utils.device import sync_device


@dataclass
class ConditionCache:
    """Cache T5 (or other) text embeddings keyed by description string."""

    _store: Dict[str, object] = field(default_factory=dict)

    def key(self, description: str) -> str:
        return hashlib.sha256(description.encode("utf-8")).hexdigest()

    def get(self, description: str):
        return self._store.get(self.key(description))

    def set(self, description: str, value) -> None:
        self._store[self.key(description)] = value


class StageInferenceEngine:
    """High-level inference API with optional compile and condition caching."""

    def __init__(
        self,
        model: LightningMusicgen,
        inference_cfg: Optional[InferenceConfig] = None,
        condition_cache: Optional[ConditionCache] = None,
    ):
        self.model = model
        self.inference_cfg = inference_cfg or InferenceConfig()
        self.condition_cache = condition_cache or ConditionCache()
        self._device = self.inference_cfg.resolved_device()
        self._dtype = resolve_dtype(self._device, self.inference_cfg.use_fp16)
        self._compiled = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        inference_cfg: Optional[InferenceConfig] = None,
    ) -> "StageInferenceEngine":
        cfg = inference_cfg or InferenceConfig()
        model = load_for_inference(checkpoint_path, inference_cfg=cfg)
        engine = cls(model, cfg)
        if cfg.compile_decode:
            engine.enable_compile()
        return engine

    @property
    def device(self) -> torch.device:
        return self._device

    def enable_compile(self) -> None:
        if self._compiled:
            return
        self.model.lm.decode_step = torch.compile(
            self.model.lm.decode_step,
            mode="reduce-overhead",
        )
        self._compiled = True

    def generate(
        self,
        n_samples: int = 1,
        gen_seconds: float = 10.0,
        prompt: Optional[Tensor] = None,
        context: Optional[Tensor | List[Tensor]] = None,
        style: Optional[Tensor] = None,
        beat: Optional[List[Beat]] = None,
        description: Optional[List[str]] = None,
        prog_bar: bool = False,
    ) -> Tensor:
        autocast_device = "cuda" if self._device.type == "cuda" else "cpu"
        if self._device.type == "mps":
            autocast_device = "mps"

        with torch.autocast(
                device_type=autocast_device,
                dtype=self._dtype,
                enabled=self.inference_cfg.use_fp16
                and self._dtype == torch.float16,
        ):
            return self.model.generate(
                n_samples=n_samples,
                gen_seconds=gen_seconds,
                prompt=prompt,
                context=context,
                style=style,
                beat=beat,
                description=description,
                prog_bar=prog_bar,
                inference_cfg=self.inference_cfg,
            )

    def benchmark(
        self,
        gen_seconds: float = 10.0,
        context: Optional[Tensor] = None,
        description: Optional[List[str]] = None,
        warmup: int = 1,
    ) -> dict:
        import time

        for _ in range(warmup):
            self.generate(
                gen_seconds=min(gen_seconds, 2.0),
                context=context,
                description=description,
            )
        sync_device(self._device)
        t0 = time.perf_counter()
        out = self.generate(
            gen_seconds=gen_seconds,
            context=context,
            description=description,
        )
        sync_device(self._device)
        elapsed = time.perf_counter() - t0
        steps = int(self.model.encodec_model.frame_rate * gen_seconds)
        audio_sec = out.shape[-1] / self.model.sample_rate
        return {
            "elapsed_s": elapsed,
            "steps_per_s": steps / elapsed,
            "rtf": elapsed / audio_sec if audio_sec > 0 else 0.0,
        }
