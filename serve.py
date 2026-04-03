"""
serve.py — TurboQuant KV-cache compressed inference server (OpenAI-compatible).

Usage:
    tq-serve --model ./models/gemma4-e4b-4bit --key-bits 4 --value-bits 4 --port 8000
    tq-serve --model google/gemma-4-E4B-it --benchmark
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import uuid
from pathlib import Path
from threading import Thread
from typing import Iterator, List

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import DynamicCache, TextIteratorStreamer

from turboquant import TurboQuantCache, load_model
from turboquant.utils import build_prompt, get_dims

_STATIC = Path(__file__).parent / "turboquant" / "static"


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="TurboQuant Inference Server", version="0.1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Server state ──────────────────────────────────────────────────────────────

_model         = None
_tokenizer     = None
_tq_args       = None
_startup_time  = time.time()
_warmed_up     = False

# Codebook dict shared across all request caches — fitted once at warmup
_shared_codebooks: dict = {}

# Running stats
_total_requests:        int   = 0
_total_tokens_generated: int  = 0
_last_tps:              float = 0.0
_last_ttft_ms:          float = 0.0
_last_kv_stats:         dict  = {}   # compression_stats() from last request


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "tq"
    messages: List[ChatMessage]
    max_tokens: int   = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float       = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/ui", include_in_schema=False)
@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
def health():
    uptime = int(time.time() - _startup_time)
    status = "ready" if _warmed_up else ("loading" if _model is None else "warming_up")

    info: dict = {
        "status":      status,
        "warmed_up":   _warmed_up,
        "uptime_s":    uptime,
        "turboquant":  True,
    }

    if _tq_args:
        info["model"]      = os.path.basename(_tq_args.model.rstrip("/\\"))
        info["key_bits"]   = _tq_args.key_bits
        info["value_bits"] = _tq_args.value_bits
        info["group_size"] = _tq_args.group_size
        bits_kv = (_tq_args.key_bits + _tq_args.value_bits) / 2
        info["theoretical_compression"] = round(16 / bits_kv, 2)

    if _last_kv_stats:
        info["kv_cache"] = {
            "compressed_MB":  round(_last_kv_stats["compressed_MB"], 3),
            "baseline_bf16_MB": round(_last_kv_stats["baseline_MB"], 3),
            "ratio":          round(_last_kv_stats["ratio"], 2),
            "note": "last request — DynamicCache would have used baseline_bf16_MB",
        }

    if torch.cuda.is_available():
        props             = torch.cuda.get_device_properties(0)
        free_b, total_b   = torch.cuda.mem_get_info(0)
        used_b            = total_b - free_b
        info["gpu"] = {
            "name":            props.name,
            "vram_total":      f"{total_b/1e9:.1f} GB",
            "vram_used":       f"{used_b/1e9:.2f} GB",
            "vram_free":       f"{free_b/1e9:.2f} GB",
            "utilization_pct": round(used_b / total_b * 100, 1),
        }

    return info


@app.get("/v1/stats")
def stats():
    uptime = int(time.time() - _startup_time)
    s: dict = {
        "uptime_s":         uptime,
        "requests_served":  _total_requests,
        "tokens_generated": _total_tokens_generated,
        "last_tps":         round(_last_tps, 1),
        "last_ttft_ms":     round(_last_ttft_ms, 1),
        "avg_tps":          round(_total_tokens_generated / max(uptime, 1), 1),
        "codebooks_cached": len(_shared_codebooks),
    }
    if torch.cuda.is_available():
        free_b, total_b   = torch.cuda.mem_get_info(0)
        s["vram_used_gb"] = round((total_b - free_b) / 1e9, 2)
        s["vram_free_gb"] = round(free_b / 1e9, 2)
    if _last_kv_stats:
        s["kv_cache"] = {
            "compressed_MB":    round(_last_kv_stats["compressed_MB"], 3),
            "baseline_bf16_MB": round(_last_kv_stats["baseline_MB"], 3),
            "ratio":            round(_last_kv_stats["ratio"], 2),
        }
    return s


@app.get("/v1/models")
def list_models():
    model_name = (
        os.path.basename(_tq_args.model.rstrip("/\\")) if _tq_args else "unknown"
    )
    meta: dict = {
        "id":            "tq",
        "object":        "model",
        "owned_by":      "local",
        "source":        _tq_args.model if _tq_args else None,
        "turboquant":    True,
        "warmed_up":     _warmed_up,
    }
    if _tq_args:
        meta["compression"] = {
            "key_bits":   _tq_args.key_bits,
            "value_bits": _tq_args.value_bits,
            "group_size": _tq_args.group_size,
        }
    if torch.cuda.is_available():
        free_b, _ = torch.cuda.mem_get_info(0)
        meta["vram_free_gb"] = round(free_b / 1e9, 2)
    return {"object": "list", "data": [meta]}


class CompareRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int   = Field(default=200, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float       = Field(default=0.95, ge=0.0, le=1.0)


@app.post("/v1/compare")
def compare(req: CompareRequest):
    """
    Run the same prompt with TurboQuantCache and DynamicCache back-to-back.
    Returns both outputs plus actual VRAM delta for each run.
    """
    if not _warmed_up:
        raise HTTPException(status_code=503, detail="Server warming up — check /health")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    msgs   = [m.model_dump() for m in req.messages]
    prompt = build_prompt(msgs, _tokenizer)

    def _alloc():
        return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # ── TurboQuant run ────────────────────────────────────────────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    alloc_before = _alloc()

    tq_cache = _make_cache(_tq_args)
    tq_text, prompt_tokens, tq_comp_tokens, tq_tps = _generate_full(
        _model, _tokenizer, tq_cache, prompt,
        req.max_tokens, req.temperature, req.top_p,
    )
    tq_vram_delta_mb = (_alloc() - alloc_before) / 1e6
    tq_stats = tq_cache.compression_stats()

    del tq_cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Baseline DynamicCache run ─────────────────────────────────────────────
    alloc_before = _alloc()
    base_result: dict = {}

    try:
        base_cache = DynamicCache()
        base_text, _, base_comp_tokens, base_tps = _generate_full(
            _model, _tokenizer, base_cache, prompt,
            req.max_tokens, req.temperature, req.top_p,
        )
        base_vram_delta_mb = (_alloc() - alloc_before) / 1e6
        del base_cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        base_result = {
            "output":            base_text,
            "completion_tokens": base_comp_tokens,
            "tps":               round(base_tps, 1),
            "kv_mb":             round(tq_stats["baseline_MB"], 3),
            "vram_delta_mb":     round(base_vram_delta_mb, 1),
            "oom":               False,
        }
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        base_result = {
            "output":    None,
            "kv_mb":     round(tq_stats["baseline_MB"], 3),
            "oom":       True,
            "oom_note":  (
                f"DynamicCache ran out of memory at {prompt_tokens} tokens — "
                f"would need {tq_stats['baseline_MB']:.1f} MB for KV cache. "
                f"TurboQuant used only {tq_stats['compressed_MB']:.1f} MB."
            ),
        }

    return {
        "prompt_tokens": prompt_tokens,
        "turboquant": {
            "output":            tq_text,
            "completion_tokens": tq_comp_tokens,
            "tps":               round(tq_tps, 1),
            "kv_compressed_mb":  round(tq_stats["compressed_MB"], 3),
            "kv_baseline_mb":    round(tq_stats["baseline_MB"], 3),
            "compression_ratio": round(tq_stats["ratio"], 2),
            "vram_delta_mb":     round(tq_vram_delta_mb, 1),
        },
        "baseline": base_result,
        "memory_saved_mb":   round(tq_stats["baseline_MB"] - tq_stats["compressed_MB"], 3),
        "compression_ratio": round(tq_stats["ratio"], 2),
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    global _total_requests, _total_tokens_generated, _last_tps, _last_ttft_ms, _last_kv_stats

    if not _warmed_up:
        raise HTTPException(
            status_code=503,
            detail="Server is warming up — codebooks not ready yet. "
                   "Check /health for status."
        )
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    msgs   = [m.model_dump() for m in req.messages]
    prompt = build_prompt(msgs, _tokenizer)
    cache  = _make_cache(_tq_args)
    req_id  = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    _total_requests += 1

    if req.stream:
        def stream_gen():
            global _total_tokens_generated, _last_tps, _last_ttft_ms
            gen_stats: dict = {}
            for tok in _generate_streaming(
                _model, _tokenizer, cache, prompt,
                req.max_tokens, req.temperature, req.top_p,
                gen_stats,
            ):
                chunk = {
                    "id": req_id, "object": "chat.completion.chunk",
                    "created": created, "model": req.model,
                    "choices": [{"index": 0,
                                 "delta": {"content": tok},
                                 "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            n    = gen_stats.get("tokens", 0)
            tps  = gen_stats.get("tps", 0.0)
            ttft = gen_stats.get("ttft_ms", 0.0)
            _total_tokens_generated += n
            _last_tps      = tps
            _last_ttft_ms  = ttft
            _last_kv_stats = cache.compression_stats()

            kv = cache.compression_stats()
            final = {
                "id": req_id, "object": "chat.completion.chunk",
                "created": created, "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": n},
                "x_tps":      round(tps, 1),
                "x_ttft_ms":  round(ttft, 1),
                "x_kv_cache": {
                    "compressed_mb": round(kv["compressed_MB"], 3),
                    "baseline_mb":   round(kv["baseline_MB"], 3),
                    "ratio":         round(kv["ratio"], 2),
                },
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    else:
        text, prompt_tokens, completion_tokens, tps = _generate_full(
            _model, _tokenizer, cache, prompt,
            req.max_tokens, req.temperature, req.top_p,
        )
        _total_tokens_generated += completion_tokens
        _last_tps      = tps
        _last_kv_stats = cache.compression_stats()

        return {
            "id": req_id, "object": "chat.completion",
            "created": created, "model": req.model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens":      prompt_tokens,
                "completion_tokens":  completion_tokens,
                "total_tokens":       prompt_tokens + completion_tokens,
            },
            "x_tps": round(tps, 1),
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_cache(args) -> TurboQuantCache:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = TurboQuantCache(
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        group_size=args.group_size,
        device=device,
        dtype=torch.bfloat16,
    )
    cache._cb = _shared_codebooks   # reuse fitted codebooks across requests
    cache.verbose = False            # suppress prints during serving
    return cache


def _warmup():
    """Pre-fit all codebooks with a short dummy forward pass at startup."""
    global _warmed_up
    print("[serve] Warming up TurboQuant codebooks (one-time)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy  = torch.randint(100, 1000, (1, 32), device=device)
    cache  = _make_cache(_tq_args)
    cache.verbose = True  # show fitting progress during warmup only
    cache._cb = _shared_codebooks
    with torch.inference_mode():
        _model(input_ids=dummy, past_key_values=cache,
               use_cache=True, return_dict=True)
    _warmed_up = True
    dims = sorted(_shared_codebooks.keys())
    print(f"[serve] Codebooks ready for dims: {dims}")


def _generate_full(
    model, tokenizer, cache, prompt,
    max_new_tokens, temperature, top_p,
) -> tuple[str, int, int, float]:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature != 1.0 or top_p < 1.0,
            use_cache=True,
            repetition_penalty=1.1,
        )
    elapsed = time.perf_counter() - t0

    completion_tokens = out.shape[1] - prompt_tokens
    tps  = completion_tokens / max(elapsed, 1e-6)
    text = tokenizer.decode(out[0][prompt_tokens:], skip_special_tokens=True)
    return text, prompt_tokens, completion_tokens, tps


def _generate_streaming(
    model, tokenizer, cache, prompt,
    max_new_tokens, temperature, top_p,
    stats: dict,
) -> Iterator[str]:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                   skip_special_tokens=True)
    kwargs = dict(
        **inputs,
        past_key_values=cache,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature != 1.0 or top_p < 1.0,
        use_cache=True,
        repetition_penalty=1.1,
    )
    Thread(target=model.generate, kwargs=kwargs).start()

    t_start     = time.perf_counter()
    token_count = 0
    for tok in streamer:
        if token_count == 0:
            stats["ttft_ms"] = (time.perf_counter() - t_start) * 1000
        token_count += 1
        yield tok

    elapsed        = time.perf_counter() - t_start
    stats["tokens"] = token_count
    stats["tps"]    = token_count / max(elapsed, 1e-6)


# ── Memory benchmark ──────────────────────────────────────────────────────────

def _run_memory_benchmark(model, tokenizer, args):
    print("\n" + "=" * 60)
    print("  TurboQuant Memory Benchmark")
    print("=" * 60)

    device = next(model.parameters()).device
    dummy  = torch.randint(100, 30000, (1, args.prompt_len), device=device)

    def vram():
        return torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0

    print(f"\n[A] Baseline DynamicCache (prompt_len={args.prompt_len})...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    v0 = vram()
    with torch.inference_mode():
        bc = DynamicCache()
        model(input_ids=dummy, past_key_values=bc, use_cache=True, return_dict=True)
    baseline_mb = vram() - v0
    print(f"  VRAM delta: {baseline_mb:.1f} MB")

    print(f"\n[B] TurboQuant (key_bits={args.key_bits}, value_bits={args.value_bits})...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    v0 = vram()
    tq = _make_cache(args)
    with torch.inference_mode():
        model(input_ids=dummy, past_key_values=tq, use_cache=True, return_dict=True)
    tq_mb  = vram() - v0
    stats  = tq.compression_stats()
    print(f"  VRAM delta:       {tq_mb:.1f} MB")
    print(f"  KV store:         {stats['compressed_MB']:.1f} MB  (packed)")
    print(f"  Baseline bf16 KV: {stats['baseline_MB']:.1f} MB")
    print(f"  Compression:      {stats['ratio']:.2f}x")
    if baseline_mb > 0:
        print(f"  VRAM ratio:       {baseline_mb/max(tq_mb,0.1):.2f}x  (measured)")

    print(f"\n[C] Key reconstruction accuracy...")
    bc2 = DynamicCache()
    with torch.inference_mode():
        model(input_ids=dummy, past_key_values=bc2, use_cache=True, return_dict=True)

    for li in list(tq._ks.keys())[:3]:
        k_tq, _ = tq._decode(li)
        try:
            kb = bc2.key_cache[li] if hasattr(bc2, "key_cache") else None
            if kb is None or kb.numel() == 0:
                continue
            cos = F.cosine_similarity(
                kb.reshape(-1, k_tq.shape[-1]).float(),
                k_tq.reshape(-1, k_tq.shape[-1]).float(), dim=-1
            ).mean().item()
            mark = "OK" if cos > 0.95 else "try more bits"
            print(f"  Layer {li:2d}: cos_sim={cos:.4f}  [{mark}]")
        except Exception as e:
            print(f"  Layer {li}: skipped ({e})")

    print("=" * 60 + "\n")


# ── Startup banner ────────────────────────────────────────────────────────────

def _print_banner(args, host: str, port: int):
    if torch.cuda.is_available():
        props          = torch.cuda.get_device_properties(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        used_b         = total_b - free_b
        gpu  = f"{props.name}  {total_b/1e9:.1f} GB VRAM"
        vram = f"{used_b/1e9:.1f} GB used  /  {free_b/1e9:.1f} GB free"
    else:
        gpu  = "CPU (no CUDA)"
        vram = "N/A"

    bits_kv = (args.key_bits + args.value_bits) / 2
    ratio   = f"~{16/bits_kv:.1f}x vs bf16"
    model_name = os.path.basename(args.model.rstrip("/\\"))

    W = 54  # inner width
    def row(s): print(f"  ║  {s:<{W}}║")

    print()
    print("  ╔" + "═" * (W + 2) + "╗")
    print(f"  ║{'TurboQuant Inference Server  v0.1.1':^{W+2}}║")
    print("  ╠" + "═" * (W + 2) + "╣")
    row(f"Model      {model_name}")
    row(f"Keys       {args.key_bits}-bit    Values  {args.value_bits}-bit    Group  {args.group_size}")
    row(f"Compression  {ratio}")
    row(f"GPU        {gpu}")
    row(f"VRAM       {vram}")
    row(f"Endpoint   http://{host}:{port}")
    print("  ╠" + "═" * (W + 2) + "╣")
    row("GET  /ui                  web interface (chat + compare)")
    row("GET  /health              server status + GPU")
    row("GET  /v1/stats            request metrics")
    row("POST /v1/chat/completions OpenAI-compatible API")
    row("POST /v1/compare          TQ vs DynamicCache side-by-side")
    print("  ╚" + "═" * (W + 2) + "╝")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="TurboQuant inference server")
    p.add_argument("--model",      default="google/gemma-4-E4B-it")
    p.add_argument("--key-bits",   type=int, default=4,
                   help="Key quantization bits: 2, 4, or 8")
    p.add_argument("--value-bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--port",       type=int, default=8000)
    p.add_argument("--host",       default="0.0.0.0")
    p.add_argument("--benchmark",  action="store_true",
                   help="Run memory + accuracy benchmark then exit")
    p.add_argument("--prompt-len", type=int, default=512,
                   help="Prompt length for --benchmark")
    return p.parse_args()


def main():
    global _model, _tokenizer, _tq_args, _startup_time
    args         = _parse_args()
    _tq_args     = args
    _startup_time = time.time()

    _model, _tokenizer = load_model(args.model)

    if args.benchmark:
        _warmup()
        _run_memory_benchmark(_model, _tokenizer, args)
        return

    _warmup()
    _print_banner(args, args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
