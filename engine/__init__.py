"""nanoserve — a minimal LLM inference engine built from first principles.

Public API:

    from engine import Engine, ModelConfig, SamplingParams
    outs = Engine(ModelConfig(model_id="Qwen/Qwen3-0.6B")).generate(["Hello"], SamplingParams(max_tokens=64))
    print(outs[0].text)
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
