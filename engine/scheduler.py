"""Request lifecycle types + a FIFO request queue.

`Sequence` holds the mutable state of one generation request; `Scheduler` is a
plain FIFO of pending sequences. Admission and eviction policy — e.g. continuous
batching, where a finished sequence frees a slot for a waiting one — belongs here,
so it can evolve without the model or the decode step changing.
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
