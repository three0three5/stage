from dataclasses import replace
from pathlib import Path
from typing import Optional
from safetensors import torch as sft

from stage.conditioning.condition_type import ConditionType
from stage.conditioning.conditioning_method import ConditioningMethod
from stage.conditioning.prompt_processor import InterleavedContextPromptProcessor
from stage.conditioning.t5embedder import T5EmbedderCPU, T5EmbedderGPU
from stage.models.lightning_musicgen import LightningMusicgen
from stage import hyperparameters as hp
from stage.inference_config import InferenceConfig, resolve_device


def _configure_attn_flash(model: LightningMusicgen, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "flash"):
            module.flash = enabled


def _configure_t5_device(model: LightningMusicgen, t5_on_cpu: bool) -> None:
    embedders = model.condition_provider.embedders
    if "description" not in embedders:
        return
    embedder = embedders["description"]
    if t5_on_cpu and hasattr(embedder, "t5"):
        embedder.t5 = embedder.t5.cpu()


def load_model(
    checkpoint_path: Path,
    device: Optional[str] = None,
    inference_cfg: Optional[InferenceConfig] = None,
) -> LightningMusicgen:
    """Load a fine-tuned STAGE checkpoint for inference."""
    inference_cfg = inference_cfg or InferenceConfig()
    if device is not None:
        inference_cfg = replace(inference_cfg, device=device)

    resolved = inference_cfg.resolved_device()
    t5_cls = T5EmbedderCPU if inference_cfg.t5_on_cpu else T5EmbedderGPU

    stage_params = hp.MusicgenParams(
        encodec_params=hp.pretrained_encodec_meta_32khz_params,
        prompt_processor_params=hp.PromptProcessorParams(
            keep_only_valid_steps=True,
            model_class=InterleavedContextPromptProcessor,
            context_dropout=0.1),
        conditioning_params=hp.ConditioningParams(
            embedder_types={
                ConditionType.DESCRIPTION: t5_cls,
            },
            conditioning_methods={
                ConditionType.DESCRIPTION: ConditioningMethod.CROSS_ATTENTION,
            },
            conditioning_dropout=0.5),
        lm_params=hp.PretrainedSmallLmParams(sep_token=2049))

    model: LightningMusicgen = stage_params.instantiate()
    sft.load_model(model, checkpoint_path)
    model = model.to(resolved).eval()
    _configure_attn_flash(model, inference_cfg.resolved_attn_flash(resolved))
    _configure_t5_device(model, inference_cfg.t5_on_cpu)
    return model


def load_for_inference(
    checkpoint_path: Path,
    inference_cfg: Optional[InferenceConfig] = None,
) -> LightningMusicgen:
    """Alias for :func:`load_model` with inference-oriented defaults."""
    return load_model(checkpoint_path, inference_cfg=inference_cfg)
