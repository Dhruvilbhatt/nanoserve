"""Request lifecycle + admission.

For experiment 0 the scheduler is intentionally trivial: a FIFO queue that hands
the engine one sequence at a time to run to completion. That is the honest
baseline — no batching yet.

Experiment 1 (continuous batching) is *this file's* job: admit up to N
sequences, form a batch each decode step from whichever are active, and evict
finished ones to admit waiting ones — all without the engine's decode step or
the model needing to change. Keeping the batch policy isolated here is the point.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from transformers import BatchEncoding

from .sampler import SamplingParams


@dataclass
class Sequence:
    """Mutable state of one generation request."""

    batch_encoding: BatchEncoding
    params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None

    def append(self, token_id: int) -> None:
        self.output_ids.append(token_id)

    def finish(self, reason: str) -> None:
        self.finish_reason = reason


@dataclass
class GenerationResult:
    """What a caller gets back for one request."""

    prompt: str
    text: str
    output_token_ids: torch.Tensor
    finish_reason: str

class Scheduler:
    """FIFO admission. Baseline runs one sequence at a time (see module docstring)."""

    def __init__(self) -> None:
        self._waiting: deque[Sequence] = deque()

    def add(self, seq: Sequence) -> None:
        self._waiting.append(seq)

    def has_waiting(self) -> bool:
        return bool(self._waiting)

    def pop(self) -> Sequence:
        return self._waiting.popleft()
