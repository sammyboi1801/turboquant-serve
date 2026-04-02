# turboquant-serve

**KV-cache memory compression for any HuggingFace model. Drop-in replacement for `DynamicCache`. No fine-tuning. No calibration data. No kernel changes.**

[![PyPI](https://img.shields.io/pypi/v/turboquant-serve)](https://pypi.org/project/turboquant-serve/)
[![Python](https://img.shields.io/pypi/pyversions/turboquant-serve)](https://pypi.org/project/turboquant-serve/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Implements [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874): random orthogonal rotation + Lloyd-Max scalar quantization applied to the KV cache at inference time.

---

## What this does

At long contexts, the KV cache becomes the memory bottleneck — not the weights. On an 8 GB GPU with a 4B model, there is almost no VRAM left for KV cache. TurboQuant compresses it 3–4× in Python, letting you run longer contexts on consumer hardware without OOM.

**What you get:**
- 3–4× reduction in KV cache memory at 4-bit keys + 4-bit values
- Same output quality for models ≥ 7B (tested: Gemma 4 E4B on RTX 4060 8 GB)
- Works with any HuggingFace model using standard `DynamicCache` — Gemma, Llama, Qwen, Mistral, Phi, DeepSeek
- OpenAI-compatible inference server — plug into Open WebUI, LiteLLM, or any OpenAI client
- Built-in web UI at `/ui` — chat, compare TQ vs baseline, live GPU stats
- `/v1/compare` endpoint: run the same prompt with TurboQuant and DynamicCache side-by-side on the already-loaded model

**What you don't get:**
- Faster tokens — dequantization happens in Python/PyTorch before attention, no FLOP reduction. Speed is similar to or slightly slower than baseline. A Triton fused kernel (roadmap) would fix this.
- Magic compatibility — models ≤ 1B or with `head_dim=64` may produce worse output at 4-bit keys; use `--key-bits 8` for those.

> **Why MSE-only, not QJL:** The TurboQuant paper describes Lloyd-Max + 1-bit QJL residual. This implementation uses Lloyd-Max MSE-only. Multiple independent community implementations found QJL hurts attention quality because softmax amplifies its variance. MSE-only wins empirically.

---

## Does it work without a GPU?

**Yes**, with caveats:

| Environment | Status |
|-------------|--------|
| CUDA GPU (recommended) | Full support — NF4 auto-quantization for VRAM < 16 GB |
| CPU only | Works — loads in float32, no bitsandbytes. Inference is slow. Use small models (≤ 1B). |
| Pre-quantized bnb checkpoint | Requires CUDA. Use a full-precision model on CPU. |

---

## Install

```bash
pip install turboquant-serve
```

PyTorch with CUDA must be installed separately (the right CUDA version matters):

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4+
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch
```

Or install from source:

```bash
git clone https://github.com/sammyboi1801/turboquant-serve
cd turboquant-serve
pip install -e .
```

---

## Downloading models from HuggingFace

Pass any HuggingFace repo ID directly — the model downloads automatically on first run and caches in `~/.cache/huggingface/`.

```bash
# Public model — no login needed
tq-serve --model Qwen/Qwen2.5-1.5B-Instruct --key-bits 8 --value-bits 4

# Gated model (Llama, Gemma) — login first
huggingface-cli login
tq-serve --model meta-llama/Llama-3.1-8B-Instruct --key-bits 4 --value-bits 4

# Local path — pre-downloaded checkpoint
tq-serve --model ./models/gemma4-e4b-4bit --key-bits 4 --value-bits 4
```

The server prints download progress. First run for a large model (e.g. 8B at bf16 = ~16 GB) takes a few minutes depending on your connection.

**Common error — gated model without login:**
```
OSError: You are trying to access a gated repo.
Fix: huggingface-cli login
```

**Common error — not enough disk space:**
```
OSError: [Errno 28] No space left on device
Fix: free disk space or set HF_HOME to a drive with more space
     HF_HOME=/mnt/data/.cache tq-serve --model ...
```

---

## Server

```bash
# Local pre-quantized checkpoint
tq-serve --model ./models/gemma4-e4b-4bit --key-bits 4 --value-bits 4 --port 8000

# Download from HuggingFace (auto NF4 on < 16 GB VRAM)
tq-serve --model google/gemma-4-E4B-it --key-bits 4 --value-bits 4

# Small model on CPU (no GPU)
tq-serve --model Qwen/Qwen2.5-0.5B-Instruct --key-bits 8 --value-bits 4

# Any Llama / Qwen / Phi / Mistral
tq-serve --model meta-llama/Llama-3.1-8B-Instruct --key-bits 4 --value-bits 4
```

On startup the server warms up codebooks with a dummy forward pass, then prints:

```
  ╔════════════════════════════════════════════════════════╗
  ║         TurboQuant Inference Server  v0.1.0           ║
  ╠════════════════════════════════════════════════════════╣
  ║  Model      gemma4-e4b-4bit                           ║
  ║  Keys       4-bit    Values  4-bit    Group  32       ║
  ║  Compression  ~4.0x vs bf16                           ║
  ║  GPU        NVIDIA GeForce RTX 4060 Laptop  8.6 GB   ║
  ║  VRAM       8.6 GB used  /  0.0 GB free               ║
  ║  Endpoint   http://0.0.0.0:8000                       ║
  ╠════════════════════════════════════════════════════════╣
  ║  GET  /ui                  web interface (chat + compare)  ║
  ║  GET  /health              server status + GPU        ║
  ║  GET  /v1/stats            request metrics            ║
  ║  POST /v1/chat/completions OpenAI-compatible API      ║
  ║  POST /v1/compare          TQ vs DynamicCache         ║
  ╚════════════════════════════════════════════════════════╝
```

Open **http://localhost:8000/ui** for the web interface.

---

## Web UI

The server ships a built-in web UI at `/ui`:

- **Chat tab** — streaming chat with live TPS counter, Stop button, KV cache stats after each message
- **Compare tab** — send one prompt, see TurboQuant and DynamicCache outputs side-by-side with memory usage
- **Stats tab** — GPU VRAM, request throughput, codebook status

No setup required — it's included in the pip package.

---

## What you can do

| Use case | How |
|----------|-----|
| Run any HF model with compressed KV cache | `tq-serve --model <repo-id-or-path>` |
| Open Web UI (chat, compare, stats) | Open `http://localhost:8000/ui` |
| Plug into Open WebUI / SillyTavern / LiteLLM | Point at `http://localhost:8000` as OpenAI provider |
| Multi-turn chat | Any OpenAI client — send full message history each turn |
| Long context without OOM | TurboQuant compresses the KV cache built per request |
| Compare TQ output vs baseline + memory | `POST /v1/compare` or Compare tab in UI |
| Memory benchmark at a given context length | `tq-serve --benchmark --prompt-len 4096` |
| Needle-in-haystack recall test | `tq-bench --model ... --lengths 1024 4096 8192` |
| Use as a Python library | `from turboquant import TurboQuantCache` |

**Multi-turn chat** works today — the server is stateless like the OpenAI API. The client sends full conversation history each turn; the server builds the KV cache fresh and compresses it. TurboQuant's benefit: conversations can be longer before OOM. A 10-turn conversation that would normally exhaust VRAM on an 8 GB GPU runs fine with TurboQuant.

---

## API reference

### `GET /health`

```json
{
  "status": "ready",
  "warmed_up": true,
  "uptime_s": 120,
  "model": "gemma4-e4b-4bit",
  "key_bits": 4,
  "value_bits": 4,
  "theoretical_compression": 4.0,
  "kv_cache": {
    "compressed_MB": 0.057,
    "baseline_bf16_MB": 0.201,
    "ratio": 3.53
  },
  "gpu": {
    "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "vram_total": "8.6 GB",
    "vram_used": "8.59 GB",
    "vram_free": "0.01 GB",
    "utilization_pct": 99.9
  }
}
```

### `GET /v1/stats`

```json
{
  "uptime_s": 300,
  "requests_served": 12,
  "tokens_generated": 840,
  "last_tps": 18.4,
  "last_ttft_ms": 312.0,
  "avg_tps": 2.8,
  "codebooks_cached": 2,
  "vram_used_gb": 8.59,
  "vram_free_gb": 0.01,
  "kv_cache": {
    "compressed_MB": 0.057,
    "baseline_bf16_MB": 0.201,
    "ratio": 3.53
  }
}
```

### `POST /v1/chat/completions`

Standard OpenAI format. Non-streaming response adds `x_tps`. Streaming final chunk adds `x_tps`, `x_ttft_ms`, and `x_kv_cache`.

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 200}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 200, "stream": true}'
```

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 47, "total_tokens": 59},
  "x_tps": 18.4
}
```

### `POST /v1/compare`

Run the same prompt with TurboQuant and DynamicCache back-to-back on the already-loaded model. No double model load — safe on 8 GB VRAM.

```bash
curl http://localhost:8000/v1/compare \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain entropy in thermodynamics."}], "max_tokens": 300}'
```

```json
{
  "prompt_tokens": 16,
  "turboquant": {
    "output": "Entropy is a measure of disorder...",
    "completion_tokens": 47,
    "tps": 18.4,
    "kv_compressed_mb": 0.057,
    "kv_baseline_mb": 0.201,
    "compression_ratio": 3.53,
    "vram_delta_mb": 12.4
  },
  "baseline": {
    "output": "Entropy is a measure of disorder...",
    "completion_tokens": 47,
    "tps": 21.1,
    "kv_mb": 0.201,
    "vram_delta_mb": 43.6
  },
  "memory_saved_mb": 0.144,
  "compression_ratio": 3.53
}
```

---

## Open WebUI / LiteLLM

Point any OpenAI-compatible client at `http://localhost:8000`:

**Open WebUI:** Settings → Connections → OpenAI API → URL: `http://localhost:8000`

**Python (openai SDK):**
```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="tq",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

---

## Python library

```python
from turboquant import TurboQuantCache
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

cache = TurboQuantCache(key_bits=4, value_bits=4)
inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)
out = model.generate(**inputs, past_key_values=cache, max_new_tokens=200)

print(cache.compression_stats())
# {'compressed_MB': 0.5, 'baseline_MB': 1.6, 'ratio': 3.53, 'key_bits': 4}
```

---

## CLI tools

| Command | What it does |
|---------|-------------|
| `tq-serve` | OpenAI-compatible inference server with web UI |
| `tq-compare` | CLI quality comparison: TQ vs DynamicCache (loads model twice — use `/v1/compare` for large models) |
| `tq-bench` | Needle-in-haystack recall at increasing context lengths |

```bash
# Needle-in-haystack
tq-bench --model ./models/gemma4-e4b-4bit --lengths 512 1024 4096 8192

# Memory benchmark at a given context length
tq-serve --model ./models/gemma4-e4b-4bit --benchmark --prompt-len 2048
```

> **Note on `tq-compare`:** Loads the model twice — will OOM on large models (≥ 4B) on 8 GB GPU. Use the server's `/v1/compare` endpoint instead.

---

## Bit config guide

| Model size | Recommended | Notes |
|------------|-------------|-------|
| ≥ 13B | `--key-bits 4 --value-bits 4` | Full quality at ~3.5× compression |
| 7B–13B | `--key-bits 4 --value-bits 4` | Tested, works well |
| 1B–7B | `--key-bits 8 --value-bits 4` | `head_dim=64` models need 8-bit keys |
| < 1B | `--key-bits 8 --value-bits 8` | Too little redundancy at 4-bit |

---

## Publishing to PyPI (auto on GitHub release)

The repo includes a GitHub Actions workflow that publishes to PyPI automatically whenever you create a GitHub Release. Uses [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/) — no API token needed.

**One-time setup on PyPI:**
1. Go to [pypi.org → Your account → Publishing](https://pypi.org/manage/account/publishing/)
2. Add a new trusted publisher:
   - Owner: `PrismML`
   - Repository: `turboquant-serve`
   - Workflow: `publish.yml`
   - Environment: `pypi`

**To publish a new version:**
1. Bump the version in `pyproject.toml` and `turboquant/__init__.py`
2. Commit and push
3. Create a GitHub Release — PyPI publish runs automatically

---

## Architecture

```
TurboQuantCache (subclass of DynamicCache)
│
├── update(k, v, layer_idx)          ← called by transformers on every forward pass
│   ├── rotate: k, v = k @ R, v @ R  ← random orthogonal matrix, seeded by head_dim
│   ├── keys:   normalize → Lloyd-Max encode → bit-pack → store uint8
│   ├── values: group min/max quant  → bit-pack → store uint8
│   └── return _decode()             ← dequantized k, v for attention
│
└── _decode(layer_idx)
    ├── unpack + Lloyd-Max decode → restore magnitude → k @ R.T
    └── group dequantize → v @ R.T
```

**Rotation:** A fixed random orthogonal matrix (seeded by `head_dim`) spreads energy evenly across dimensions before quantization, reducing worst-case error vs. quantizing raw activations.

**Keys vs values:** Keys are normalized to the unit sphere then Lloyd-Max quantized (MSE-optimal for the Beta distribution prior on rotated unit vectors). Values use group-wise affine quantization (min/max per group of 32).

**Codebook sharing:** Codebooks are fitted once at server startup via a warmup forward pass and shared across all requests. Re-fitting per request would add ~1s latency.

**Why no QJL:** QJL provides unbiased inner product estimates but introduces variance. Softmax amplifies this variance exponentially. MSE-only consistently achieves better end-task quality — confirmed by multiple independent implementations.

---

## Roadmap

- [ ] Triton fused dequant+attention kernel — compute attention directly on quantized K/V without materializing bf16 (true memory bandwidth + FLOP reduction)
- [ ] Residual window — keep last N tokens in fp16 for recency quality
- [ ] Outlier-aware mixed precision — more bits for outlier channels
- [ ] Prefix caching — reuse KV cache across requests with shared prefixes
- [ ] vLLM integration
- [ ] Multi-GPU / tensor parallel support

---

## Inspiration and related work

This project implements the algorithm from:

> **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**  
> Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni  
> ICLR 2026 · [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

The server architecture (OpenAI-compatible API, streaming, warmup, codebook caching) is inspired by [llama.cpp server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server) and [Ollama](https://github.com/ollama/ollama). The web UI design follows llama.cpp's minimal dark aesthetic.

Related community implementations of TurboQuant:
- [0xSero/turboquant](https://github.com/0xSero/turboquant) — vLLM + Triton kernels
- [back2matching/turboquant](https://github.com/back2matching/turboquant) — pip package
- [scos-lab/turboquant](https://github.com/scos-lab/turboquant) — research analysis
- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) — llama.cpp / Metal

---

## Citation

```bibtex
@inproceedings{zandieh2026turboquant,
  title     = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author    = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle = {ICLR},
  year      = {2026}
}
```
