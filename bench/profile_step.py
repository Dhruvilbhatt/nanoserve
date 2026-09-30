#!/usr/bin/env python3
"""Driver to profile ONE batch-1 generate under nsys — exposes the per-step
GPU-idle gaps (GPU waiting on the Python decode loop between kernel launches).

Profiles only the region between cudaProfilerStart/Stop (the second generate),
so model load + warmup are excluded. Run via:

  nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
      --capture-range-end=stop --stats=true -o results/decode_profile \
      --force-overwrite true python bench/profile_step.py
"""
from __future__ import annotations

import time

import torch

from engine import Engine, ModelConfig, SamplingParams

N = 16  # decode steps to profile


def main() -> None:
    eng = Engine(ModelConfig(model_id="Qwen/Qwen3-0.6B"))
    params = SamplingParams(max_tokens=N)
    eng.generate(["The capital of France is"], params)  # warmup: model load + caches
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push("decode_region")
    t0 = time.perf_counter()
    eng.generate(["The capital of France is"], params)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()

    print(f"[profile] batch=1, {N} tokens: wall {dt*1e3:.1f} ms  ->  {dt*1e3/N:.1f} ms/token")


if __name__ == "__main__":
    main()
