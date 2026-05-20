# Stage - Single Stem Accompaniment Generation
[![ArXiv](https://img.shields.io/badge/arXiv-2504.05690-b31b1b?logo=arxiv&labelColor=gray)](https://arxiv.org/abs/2504.05690)
[![demo](https://img.shields.io/badge/Demo-Samples-1997B5?labelColor=gray)](https://https://giorgioskij.github.io/stage-demo/)


### Disclaimer:
This is very much **research-oriented** code, it's not by any means production-ready. You'll see configurations for many failed experiments, and some seemingly overcomplicated structures that were necessary for our testing and experimentation workflow. 

If you are interested in making this code more usable, feel free to contribute or to ask questions! <br>Research is only beautiful when it's shared.


# Prerequisites
To run code in this repo, you need to:

- ## Set up the environment:
    This repo's environment is managed by [uv](https://docs.astral.sh/uv/).

    Setting everything up should be as easy as:
    - cloning the repo
    - `cd`-ing into it
    - running `uv sync`

- ## Download the weights for the pre-trained components of our models
    For ease of use, we entirely rewrote `MusicGen`'s architecture, so you'll need to download pre-trained weights that are compatible with our model.

    - Download the weights [here](https://drive.google.com/drive/folders/1tNZsN3OYC8PGVklCRPkebPvS5QGX1w1q?usp=sharing) and place them in the `weights/` directory of this repo. You can place them anywhere else if you'd like, but modify `src/stage/config.py` accordingly if you do so.

    - If you want to run inference on our trained models, download the fine-tuned checkpoints [here](https://drive.google.com/drive/folders/1tNZsN3OYC8PGVklCRPkebPvS5QGX1w1q?usp=sharing) and place them inside the `checkpoints/` directory.


# Inference

Follow the example in `src/stage/inference.py` to test inference with any model.

## Local inference (optimized path)

The repo includes an inference-focused runtime with **KV-cache autoregressive decoding**, automatic device selection (`cuda` → `mps` → `cpu`), and optional FP16.

```python
from stage.inference_config import InferenceConfig
from stage.runtime.inference_engine import StageInferenceEngine

cfg = InferenceConfig(
    use_kv_cache=True,      # prefill context + incremental decode (recommended)
    use_fp16=True,          # FP16 on CUDA/MPS; stays FP32 on CPU
    t5_on_cpu=True,         # keep T5 on CPU when the LM uses MPS/CUDA
    compile_decode=False,   # torch.compile on decode_step (experimental)
    device=None,            # None = auto; or "cpu", "cuda", "mps"
)
engine = StageInferenceEngine.from_checkpoint("checkpoints/stage-drums.safetensors", cfg)
audio = engine.generate(n_samples=1, gen_seconds=10, context=wav, description=["heavy rock drums"])
```

Pass the same `InferenceConfig` to `LightningMusicgen.generate(..., inference_cfg=cfg)` if you load the model directly.

### `InferenceConfig` fields

| Field | Default | Description |
|-------|---------|-------------|
| `cfg_coef` | `3.0` | Classifier-free guidance scale (cond vs uncond logits). |
| `top_k` | `250` | Top-k sampling for the next codebook token. |
| `use_kv_cache` | `True` | Prefill the interleaved prefix once, then one `decode_step` per new timestep. Much faster than full-prefix forward each step. |
| `use_fp16` | `True` | Half precision on GPU/MPS; ignored on CPU (always FP32). |
| `compile_decode` | `False` | Wrap `MusicgenLm.decode_step` with `torch.compile` (inference engine only). |
| `t5_on_cpu` | `True` | Run the T5 text encoder on CPU; set `False` only if your checkpoint loads T5 on the same device as the LM. |
| `device` | `None` | Force device string, or `None` for automatic selection. |
| `attn_flash` | `None` | Flash attention in the LM decoder: `None` = on except MPS, `True`/`False` to override. |
| `greedy` | `False` | Argmax instead of top-k (for deterministic tests, not typical listening). |
| `verify_kv_parity` | `False` | **Debug only:** each KV step also runs a full-prefix forward and resyncs the cache on mismatch. Correct but much slower; leave `False` for production. |

KV decoding applies a **key padding mask** (`self_attn_kv_mask`) on cached attention so interleaved sequences with invalid conditioning timesteps match the masked full forward pass.

### Benchmark CLI

Wall time, steps/s, and real-time factor (RTF):

```bash
python -m stage.benchmark_inference --checkpoint checkpoints/stage-drums.safetensors --gen-seconds 10
```

Compare KV on vs off (writes `bass_no_kv_*` / `bass_kv_*` under `--output-dir`):

```bash
python -m stage.benchmark_inference --compare-kv --gen-seconds 1 --output-dir outputs/local_inference
```

Useful flags:

| Flag | Description |
|------|-------------|
| `--checkpoint` | Path to `.safetensors` checkpoint (default: `checkpoints/stage-bass.safetensors`). |
| `--audio` | Context WAV/MP3 for accompaniment generation. |
| `--context-seconds` | Trim context to the first N seconds. |
| `--gen-seconds` | Length of generated audio in seconds. |
| `--output` / `--output-dir` | Save a single WAV or named files per run. |
| `--compare-kv` | Run once with `use_kv_cache=False`, then `True`. |
| `--no-kv-cache` | Disable KV cache for a single run. |
| `--no-fp16` | Force FP32. |
| `--no-attn-flash` / `--attn-flash` | Disable or force flash attention. |
| `--t5-on-gpu` | Load T5 on the inference device (required for some bass checkpoints on CPU). |
| `--device` | `cpu`, `cuda`, or `mps`. |
| `--null-description` | Use `description=[None]` (no text conditioning). |
| `--warmup` | Warmup generations before timing (default: 1). |

Example (CPU, short generation, Kaggle-style null description):

```bash
python -m stage.benchmark_inference \
  --checkpoint checkpoints/stage-bass.safetensors \
  --audio path/to/context.mp3 \
  --context-seconds 2 \
  --gen-seconds 1 \
  --compare-kv \
  --null-description \
  --device cpu \
  --no-fp16 \
  --no-attn-flash \
  --t5-on-gpu \
  --warmup 0 \
  --output-dir outputs/local_inference
```

### Tests

KV-cache parity (tiny LM, no downloaded weights):

```bash
python -m pytest tests/test_kv_cache_parity.py tests/test_kv_cache_generate_parity.py -v
```

Micro-benchmark on the tiny LM:

```bash
python tests/benchmark_kv_cache_lm.py
```


# Data:

The full fine-tuning, validation, and testing is performed on splits of the MoisesDB dataset.

The dataset should be kept in any of the folders listed in `stage.config.moises_path()`, such as `stage/datasets/moisesdb`.

Data preprocessing is needed to train the model. For each song:
- all the drums tracks should be mixed into a single track;
- a `features.json` file should be generated containing features extracted with *essentia*.
This can be done with the function `stage.data.StemmedDataset.prepare_data()`

The structure of the dataset directory should look something like:
```
moisesdb
| 014f3712-293b-42af-9f29-0ed1785be792 
    | features.json
    | bass
    |   | 47c825c0-1c9d-46ec-902c-0037fa45ec54.wav
    | drums_mixed
    |   | drums.wav
    | guitar/
    | ...
| ...
```

# Training: 

The model can be trained using the training script in `src/stage/train_stage_drums.py`
Here you can set all the hyperparameters for both the dataset and the model.

Runs are by default logged to Weights And Biases. Make sure to either:
- set your WandB entity/project name in `stage/config.py`; or
- set `log=False` in the `train(...)` function call
