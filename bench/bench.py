#!/usr/bin/env python3
"""Benchmark one engine variant on a fixed prompt set.

Records each prompt's output token ids + throughput to a JSON receipt, and can
(a) verify outputs against HF greedy and (b) diff ids + report speedup vs a saved
baseline. It only calls the public `Engine.generate(prompts, params)` API, so the
same harness runs unchanged across the baseline (sequential) and batched commits.

  python bench/bench.py --label baseline --out results/baseline.json --check-hf
  python bench/bench.py --label batched  --out results/batched.json \
      --check-hf --compare results/baseline.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from engine import Engine, ModelConfig, SamplingParams

PROMPTS = [
    "The capital of France is",
    "Hello",
    "def fibonacci(n):",
    "Once upon a time",
]


def _to_ids(result) -> list[int]:
    x = result.output_token_ids
    return x.tolist() if isinstance(x, torch.Tensor) else list(x)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _prefix_match(a: list[int], b: list[int]) -> bool:
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None, help="baseline JSON to diff ids + speedup against")
    ap.add_argument("--check-hf", action="store_true")
    args = ap.parse_args()

    engine = Engine(ModelConfig(model_id=args.model))
    params = SamplingParams(max_tokens=args.max_tokens)

    engine.generate(PROMPTS, params)  # warmup (compile/caches/clocks)
    _sync()
    t0 = time.perf_counter()
    results = engine.generate(PROMPTS, params)
    _sync()
    elapsed = time.perf_counter() - t0

    out_ids = [_to_ids(r) for r in results]
    total = sum(len(x) for x in out_ids)
    tps = total / elapsed
    device = "cuda" if next(engine.model.model.parameters()).is_cuda else "cpu"

    record = {
        "label": args.label, "model": args.model, "device": device,
        "max_tokens": args.max_tokens, "prompts": PROMPTS,
        "output_token_ids": out_ids,
        "wall_s": elapsed, "total_tokens": total, "tokens_per_s": tps,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"[{args.label}] device={device}  {total} tok in {elapsed:.3f}s  ->  {tps:.1f} tok/s")

    if args.check_hf:
        model, tok = engine.model.model, engine.model.tokenizer
        dev = next(model.parameters()).device
        ok = True
        for p, ids in zip(PROMPTS, out_ids):
            enc = tok(p, return_tensors="pt")["input_ids"].to(dev)
            with torch.inference_mode():
                hf = model.generate(enc, max_new_tokens=args.max_tokens,
                                    do_sample=False)[0, enc.shape[1]:].tolist()
            m = _prefix_match(ids, hf)
            ok &= m
            print(f"  HF {'OK  ' if m else 'DIFF'} {p!r}")
        print("HF correctness:", "PASS" if ok else "FAIL")

    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        parity = all(_prefix_match(a, b) for a, b in zip(out_ids, base["output_token_ids"]))
        speedup = tps / base["tokens_per_s"] if base["tokens_per_s"] else float("nan")
        print(f"vs '{base['label']}' ({base['device']}): "
              f"outputs {'MATCH' if parity else 'DIFFER'} | "
              f"{base['tokens_per_s']:.1f} -> {tps:.1f} tok/s = {speedup:.2f}x")


if __name__ == "__main__":
    main()
