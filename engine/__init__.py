"""mini-serve engine — a minimal LLM serving engine used as a lab bench.

Public API:

    from engine import Engine, ModelConfig, SamplingParams
    out = Engine(ModelConfig(model_id="Qwen/Qwen3-0.6B")).generate("Hello", SamplingParams(max_tokens=64))
    print(out.text)
"""
from __future__ import annotations

from .model import ModelConfig, ModelRunner
from .sampler import SamplingParams, sample
from .scheduler import GenerationResult, Sequence
from .server import Engine

__all__ = [
    "Engine",
    "ModelConfig",
    "ModelRunner",
    "SamplingParams",
    "sample",
    "GenerationResult",
    "Sequence",
]
