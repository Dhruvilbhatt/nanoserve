"""Op-level backend selection — the seam where kernel experiments attach.

Baseline: everything is "torch" (stock HuggingFace ops) and the engine runs the
model unchanged. Experiment 4 (the hosted kernels, from the sibling gpu-kernels
repo) will register custom attention / rmsnorm / gemm implementations here and
have ModelRunner route the chosen ops to them — so swapping a kernel in is one
config flag, not a fork of the model code.

Kept deliberately tiny until there is a real kernel to register; this exists now
only to pin down the seam and fail loudly on a bad backend name.
"""
from __future__ import annotations

# Add "triton" / "cuda" here once kernels/ is wired in (experiment 4).
KNOWN_BACKENDS = ("torch",)


def resolve(name: str) -> str:
    if name not in KNOWN_BACKENDS:
        raise ValueError(f"unknown backend {name!r}; known: {KNOWN_BACKENDS}")
    return name
