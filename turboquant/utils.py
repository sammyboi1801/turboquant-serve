"""
turboquant/utils.py — Model loading and config helpers.

Supports:
  - Pre-quantized bitsandbytes checkpoints (requires CUDA)
  - Full-precision HuggingFace models (auto-quantized NF4 on small VRAM GPUs)
  - CPU-only inference (full-precision only, slow but functional)
  - Any HuggingFace model ID or local path

To download a model from HuggingFace:
  - Public models: just pass the repo ID, e.g. "meta-llama/Llama-3.2-1B-Instruct"
  - Gated models (Llama, Gemma): run `huggingface-cli login` first
"""

from __future__ import annotations
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, force_4bit: bool = False):
    """
    Load any HuggingFace causal LM for inference.

    Behaviour by environment:
      - CUDA GPU < 16 GB VRAM  →  NF4 on-the-fly quantization (bitsandbytes)
      - CUDA GPU ≥ 16 GB VRAM  →  bfloat16, no quantization
      - CPU only               →  float32, no quantization (slow — use small models)

    Pre-quantized bitsandbytes checkpoints are loaded as-is and require CUDA.
    Passing a HuggingFace repo ID will download the model automatically on first run
    and cache it in ~/.cache/huggingface/. Use `huggingface-cli login` for gated models.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[load] Tokenizer: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    already_4bit = _detect_bnb_4bit(model_id)
    print(f"[load] Model: {model_id}  (pre-quantized={already_4bit})")

    if device == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[load] GPU: {torch.cuda.get_device_name(0)}  VRAM: {total_gb:.1f} GB")
    else:
        print(
            "[load] No CUDA GPU detected — running on CPU.\n"
            "         Inference will be slow. For best results use a GPU.\n"
            "         Only full-precision models are supported on CPU\n"
            "         (bitsandbytes NF4 quantization requires CUDA)."
        )

    if already_4bit:
        if device != "cuda":
            raise RuntimeError(
                "Pre-quantized bitsandbytes checkpoint requires a CUDA GPU.\n"
                "  Either use a GPU, or download a full-precision model instead.\n"
                f"  Model: {model_id}"
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda:0", low_cpu_mem_usage=True
        )
    elif device == "cuda" and (force_4bit or _should_quantize()):
        print("[load] Applying NF4 on-the-fly quantization (VRAM < 16 GB)…")
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
    else:
        # GPU with ≥ 16 GB, or CPU
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    model.eval()
    devices = {str(p.device) for p in model.parameters()}
    print(f"[load] Loaded on: {devices}")
    return model, tok


def get_dims(model) -> tuple[int, int, int, int]:
    """Return (n_heads, n_kv_heads, head_dim, n_layers) from model config."""
    cfg = model.config
    tc = getattr(cfg, "text_config", cfg)
    n_heads  = tc.num_attention_heads
    n_kv     = getattr(tc, "num_key_value_heads", n_heads)
    hidden   = tc.hidden_size
    head_dim = getattr(tc, "head_dim", hidden // n_heads)
    n_layers = tc.num_hidden_layers
    return n_heads, n_kv, head_dim, n_layers


def build_prompt(messages: list[dict], tokenizer) -> str:
    """Apply chat template or fall back to a simple role-tagged format."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    # Fallback: Gemma-style turn tags
    parts = [
        f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>"
        for m in messages
    ]
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_bnb_4bit(model_id: str) -> bool:
    """Check if a local checkpoint has a bitsandbytes quantization config."""
    cfg_path = os.path.join(model_id, "config.json") if os.path.isdir(model_id) else None
    if not cfg_path or not os.path.exists(cfg_path):
        return False
    with open(cfg_path) as f:
        qcfg = json.load(f).get("quantization_config", {})
    return (
        qcfg.get("quant_type") in ("nf4", "fp4")
        or qcfg.get("quant_method") == "bitsandbytes"
        or qcfg.get("load_in_4bit") is True
    )


def _should_quantize() -> bool:
    """Apply NF4 quantization when GPU VRAM < 16 GB (covers most consumer cards)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).total_memory / 1e9 < 16.0
