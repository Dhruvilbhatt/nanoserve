"""KV cache — baseline delegates to HF's DynamicCache.

DynamicCache is a contiguous, per-sequence cache that grows one token per decode
step. That is the correct, unremarkable baseline. Experiment 2 (paged KV) will
replace this factory with a cache that allocates fixed-size blocks from a shared
pool so many sequences share memory without per-sequence over-allocation.

The whole seam is this one factory: the engine only ever calls `new_cache()`, so
the paged version slots in without touching the decode loop.
"""
from __future__ import annotations

from transformers import DynamicCache


def new_cache() -> DynamicCache:
    """A fresh, empty KV cache for one sequence."""
    return DynamicCache()
