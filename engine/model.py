"""ModelRunner: loads one HF causal-LM + tokenizer and runs a single forward.

This is the torch-op baseline: the forward pass is stock HuggingFace, so the
engine stands up without any kernel work. Kernel experiments attach at
`backends.py`; this class is the only place that touches the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import backends


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen3-8B"   # target; override for smoke tests
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    backend: str = "torch"


class ModelRunner:
    def __init__(self, config: ModelConfig):
        self.config = config
        backends.resolve(config.backend)  # validate up front; fail loud on typos

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(config.model_id, dtype=config.dtype)
            # .to(config.device)   # (moving to the GPU is a later experiment)
            # .eval()
        )
        self.eos_token_ids = self._resolve_eos()

    def _resolve_eos(self) -> set[int]:
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        if eos is None:
            return set()
        return {eos} if isinstance(eos, int) else set(eos)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def forward(self, input_ids, cache):
        """One forward pass. `input_ids` is [1, seq_len] — the full prompt on
        prefill, a single token on each decode step. Returns (logits, cache).
        """
        out = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        return out.logits, out.past_key_values
