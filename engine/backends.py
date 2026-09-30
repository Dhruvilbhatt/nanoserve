"""Op-level backend selection — the seam where custom kernels would attach.

Today everything is "torch" (stock HuggingFace ops) and the engine runs the model
unchanged. Custom attention / rmsnorm / gemm implementations could register here
and have ModelRunner route the chosen ops to them, so swapping a kernel in is one
config flag rather than a fork of the model code.

Kept deliberately tiny until there is a real kernel to register; for now it just
pins down the seam and fails loudly on a bad backend name.
"""
from __future__ import annotations

# Add "triton" / "cuda" here once custom kernels are wired in.
KNOWN_BACKENDS = ("torch",)


def resolve(name: str) -> str:
    if name not in KNOWN_BACKENDS:
        raise ValueError(f"unknown backend {name!r}; known: {KNOWN_BACKENDS}")
    return name
