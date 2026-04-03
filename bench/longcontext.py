"""
bench/longcontext.py: Needle-in-haystack test at increasing context lengths.

Tests whether TurboQuant preserves recall of a hidden fact buried deep in
a long document, at multiple context lengths. Mirrors the evaluation used
in the TurboQuant paper (§4.2).

Usage:
  python bench/longcontext.py --model ./models/gemma4-e4b-4bit
  python bench/longcontext.py --model ./models/gemma4-e4b-4bit \
      --lengths 1024 4096 8192 16384 --key-bits 4 --value-bits 4
"""

import argparse

import torch
from transformers import DynamicCache
from turboquant import TurboQuantCache
from turboquant.utils import load_model

NEEDLE = "The secret project code name is AURORA-7749"
QUESTION = "What is the secret project code name mentioned in the document?"
FILLER = (
    "The development of large language models has accelerated significantly "
    "over the past few years. Researchers have proposed numerous techniques "
    "to improve efficiency, including quantization, pruning, and distillation. "
    "These methods aim to reduce the computational cost of inference while "
    "maintaining model quality. The key challenge is balancing compression "
    "ratio against accuracy degradation. "
)


def make_haystack(tokenizer, context_len: int, needle_depth: float = 0.5) -> str:
    """Build a document of ~context_len tokens with needle at depth position."""
    filler_toks = len(tokenizer.encode(FILLER))
    total_filler = int(context_len * 0.9)
    repeats = total_filler // filler_toks + 1
    filler = (FILLER * repeats)

    # Split at needle_depth and insert needle
    words = filler.split()
    insert_at = int(len(words) * needle_depth)
    words.insert(insert_at, f"\n\n{NEEDLE}\n\n")
    return " ".join(words)


def run_niah(model, tokenizer, context_len: int, cache, label: str,
             needle_depth: float = 0.5) -> dict:
    device = next(model.parameters()).device
    haystack = make_haystack(tokenizer, context_len, needle_depth)
    prompt = (
        f"Read the following document carefully:\n\n{haystack}\n\n"
        f"Question: {QUESTION}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       max_length=context_len + 200,
                       truncation=True).to(device)
    actual_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
        )

    answer = tokenizer.decode(out[0][actual_len:], skip_special_tokens=True).strip()
    found = "AURORA-7749" in answer or "aurora" in answer.lower()

    return {
        "label":       label,
        "context_len": actual_len,
        "found":       found,
        "answer":      answer[:100],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True)
    p.add_argument("--lengths",     type=int, nargs="+",
                   default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--key-bits",    type=int, default=4)
    p.add_argument("--value-bits",  type=int, default=4)
    p.add_argument("--depths",      type=float, nargs="+",
                   default=[0.25, 0.5, 0.75],
                   help="Needle insertion depths (0=start, 1=end)")
    args = p.parse_args()

    model, tokenizer = load_model(args.model)

    print(f"\n{'─'*65}")
    print(f"  Needle-in-Haystack  |  key={args.key_bits}-bit  val={args.value_bits}-bit")
    print(f"{'─'*65}")
    print(f"  {'Context':>8}  {'Depth':>6}  {'Baseline':>10}  {'TurboQuant':>12}")
    print(f"{'─'*65}")

    for ctx_len in args.lengths:
        for depth in args.depths:
            # Baseline
            try:
                r_base = run_niah(model, tokenizer, ctx_len,
                                  DynamicCache(), "Baseline", depth)
                base_str = "✓ FOUND" if r_base["found"] else "✗ MISSED"
                base_ctx = r_base["context_len"]
            except torch.cuda.OutOfMemoryError:
                base_str = "OOM"
                base_ctx = ctx_len
                torch.cuda.empty_cache()

            # TurboQuant
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tq_cache = TurboQuantCache(key_bits=args.key_bits,
                                           value_bits=args.value_bits,
                                           device=device)
                r_tq = run_niah(model, tokenizer, ctx_len,
                                tq_cache, "TurboQuant", depth)
                tq_str = "✓ FOUND" if r_tq["found"] else "✗ MISSED"
            except torch.cuda.OutOfMemoryError:
                tq_str = "OOM"
                torch.cuda.empty_cache()

            print(f"  {base_ctx:>8,}  {depth:>5.0%}  {base_str:>10}  {tq_str:>12}")

    print(f"{'─'*65}\n")


if __name__ == "__main__":
    main()