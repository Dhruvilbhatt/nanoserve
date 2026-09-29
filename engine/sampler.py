"""Token selection from next-token logits.

Greedy (temperature == 0) is the default and is what the correctness check
compares against HF's `do_sample=False`. Temperature / top-k / top-p live here
too because sampling is control-plane work, not a kernel experiment.

`sample()` operates on a batch of last-token logits (shape [N, vocab]) and
returns the chosen token ids (shape [N, 1]).
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
           generator: torch.Generator | None = None) -> torch.Tensor:
    """Pick the next token id per row from `logits` ([N, vocab]) -> [N, 1]."""
    if params.greedy:
        return logits.argmax(dim=-1).unsqueeze(-1)

    logits = logits / params.temperature
    logits = _top_k(logits, params.top_k)
    logits = _top_p(logits, params.top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return logits
    kth = torch.topk(logits, k).values[:, -1].unsqueeze(-1)
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
    return logits.scatter(0, idx, ordered, dim=1)
