"""Token selection from next-token logits (single sequence).

Greedy (temperature == 0) is the default and is what the correctness check
compares against HF's `do_sample=False`. `sample()` operates on one sequence's
last-token logits (shape [vocab]) and returns an int token id.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0   # 0.0 => greedy (argmax)
    top_k: int = 0             # 0 => disabled
    top_p: float = 1.0         # 1.0 => disabled
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def sample(logits: torch.Tensor, params: SamplingParams,
           generator: torch.Generator | None = None) -> int:
    """Pick the next token id from `logits` ([vocab])."""
    if params.greedy:
        return int(logits.argmax())

    logits = logits / params.temperature
    logits = _top_k(logits, params.top_k)
    logits = _top_p(logits, params.top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


def _top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return logits
    kth = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < kth, float("-inf"))


def _top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if not 0.0 < p < 1.0:
        return logits
    ordered, idx = torch.sort(logits, descending=True)
    probs = torch.softmax(ordered, dim=-1)
    # Drop tokens once the cumulative prob *before* them already exceeds p — this
    # keeps the smallest set whose mass exceeds p, and always keeps the top token.
    drop = (probs.cumsum(dim=-1) - probs) > p
    ordered = ordered.masked_fill(drop, float("-inf"))
    return logits.scatter(0, idx, ordered)
