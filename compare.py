"""
compare.py — Side-by-side quality comparison: TurboQuant vs standard KV cache.

Usage:
    tq-compare --model google/gemma-4-E4B-it --prompt "Explain quantum entanglement"
    tq-compare --model ./models/gemma-4-e4b-4bit --key-bits 2 --value-bits 4
"""

from __future__ import annotations

import argparse
import gc
import textwrap

import torch
from transformers import DynamicCache

from turboquant import TurboQuantCache, load_model
from turboquant.utils import build_prompt


def _generate(model, tokenizer, cache, prompt: str, max_new_tokens: int,
              temperature: float, top_p: float) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)


def run_compare(model, tokenizer, args):
    messages = [{"role": "user", "content": args.prompt}]
    prompt = build_prompt(messages, tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print(f"  Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"  max_new_tokens={args.max_tokens}  key_bits={args.key_bits}  "
          f"value_bits={args.value_bits}  group_size={args.group_size}")
    print("=" * 70)

    # Standard cache
    print("\n[A] Standard DynamicCache:")
    print("-" * 40)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    baseline_out = _generate(
        model, tokenizer, DynamicCache(), prompt,
        args.max_tokens, args.temperature, args.top_p,
    )
    for line in textwrap.wrap(baseline_out, width=68):
        print(f"  {line}")

    # TurboQuant cache
    print(f"\n[B] TurboQuant (key_bits={args.key_bits}, value_bits={args.value_bits}):")
    print("-" * 40)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tq = TurboQuantCache(
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        group_size=args.group_size,
        device=device,
        dtype=torch.bfloat16,
    )
    tq_out = _generate(
        model, tokenizer, tq, prompt,
        args.max_tokens, args.temperature, args.top_p,
    )
    for line in textwrap.wrap(tq_out, width=68):
        print(f"  {line}")

    stats = tq.compression_stats()
    print(f"\n  Compression: {stats['ratio']:.2f}x  "
          f"({stats['compressed_MB']:.1f} MB vs {stats['baseline_MB']:.1f} MB bf16)")
    print("=" * 70 + "\n")


def _parse_args():
    p = argparse.ArgumentParser(description="Compare TurboQuant vs standard KV cache")
    p.add_argument("--model",       default="google/gemma-4-E4B-it")
    p.add_argument("--prompt",      default="Explain the concept of entropy in thermodynamics.")
    p.add_argument("--key-bits",    type=int, default=4)
    p.add_argument("--value-bits",  type=int, default=4)
    p.add_argument("--group-size",  type=int, default=32)
    p.add_argument("--max-tokens",  type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p",       type=float, default=0.95)
    return p.parse_args()


def main():
    args = _parse_args()
    model, tokenizer = load_model(args.model)
    run_compare(model, tokenizer, args)


if __name__ == "__main__":
    main()
