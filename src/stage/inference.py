"""Example local inference using the optimized runtime."""

import lightning as L
import torch

from stage import config as cfg
from stage.inference_config import InferenceConfig
from stage.runtime.inference_engine import StageInferenceEngine
from stage.utils.audio import load_audio, save_audio

INSTRUMENT = "drums"
SAMPLE = "sample2"
SEED = 42

inference_cfg = InferenceConfig(
    use_kv_cache=True,
    use_fp16=True,
    t5_on_cpu=True,
    compile_decode=False,
)

checkpoint_path = cfg.CKP_DIR / f"stage-{INSTRUMENT}.safetensors"
engine = StageInferenceEngine.from_checkpoint(checkpoint_path, inference_cfg)

desc_path = cfg.AUDIO_DIR / INSTRUMENT / f"{SAMPLE}-desc.txt"
desc = desc_path.read_text().strip() if desc_path.exists() else None

wav = load_audio(cfg.AUDIO_DIR / INSTRUMENT / f"{SAMPLE}.wav").to(engine.device)

L.seed_everything(SEED)
out = engine.generate(
    n_samples=1,
    gen_seconds=10,
    prompt=None,
    context=wav,
    style=None,
    beat=None,
    description=[desc],
    prog_bar=True,
)

save_audio(out, cfg.AUDIO_DIR / "gen" / f"{SAMPLE}_{INSTRUMENT}_{SEED}.wav")
padded_wav = torch.nn.functional.pad(
    wav,
    (0, out.shape[-1] - wav.shape[-1]),
    value=0,
)
mix = out + padded_wav
save_audio(mix, cfg.AUDIO_DIR / "gen" / f"{SAMPLE}_{INSTRUMENT}_{SEED}_mix.wav")
