#!/usr/bin/env python3
"""Batch-size sweep: throughput + latency vs batch size, for one device.

Replicates a single prompt to each batch size and generates a fixed token budget,
so the only thing changing across rows is the batch size. Run once per device to
get comparable curves (the point: GPU throughput scales with batch as its
parallelism gets fed; CPU flattens because it was never the idle resource):

  python bench/sweep.py --device cuda --batch-sizes 1,2,4,8,16,32,64,128,256 --out results/sweep_cuda.csv
  python bench/sweep.py --device cpu  --batch-sizes 1,2,4,8,16,32            --out results/sweep_cpu.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from engine import Engine, ModelConfig, SamplingParams

PROMPT = "The capital of France is"


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sizes = [int(x) for x in args.batch_sizes.split(",")]
    engine = Engine(ModelConfig(model_id=args.model, device=args.device))
    params = SamplingParams(max_tokens=args.max_tokens)

    engine.generate([PROMPT], params)  # warmup (CUDA init / caches / clocks)

    rows = []
    print(f"device={args.device}  model={args.model}  max_tokens={args.max_tokens}")
    print(f"{'batch':>6} {'tok/s':>10} {'req/s':>9} {'latency_s':>11}")
    for n in sizes:
        prompts = [PROMPT] * n
        _sync()
        t0 = time.perf_counter()
        engine.generate(prompts, params)
        _sync()
        dt = time.perf_counter() - t0
        total = n * args.max_tokens
        rows.append({
            "device": args.device, "batch_size": n, "wall_s": round(dt, 4),
            "total_tokens": total, "tokens_per_s": round(total / dt, 1),
            "requests_per_s": round(n / dt, 3),
        })
        print(f"{n:>6} {total / dt:>10.1f} {n / dt:>9.2f} {dt:>11.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> wrote {args.out}")


if __name__ == "__main__":
    main()
