"""Request lifecycle + admission.

For the baseline the scheduler is intentionally trivial: a FIFO queue that hands
the engine one sequence at a time to run to completion. That is the honest
starting point — no batching yet.

Batching (the next experiment) is *this file's* job: form a batch each decode
step from whichever sequences are active, and evict finished ones — all without
the model or the decode step changing. Keeping the batch policy isolated here is
the point.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .sampler import SamplingParams


@dataclass
class Sequence:
    """Mutable state of one generation request."""

    prompt_ids: list[int]
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
    output_token_ids: list[int]
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
