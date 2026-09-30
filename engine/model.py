"""ModelRunner: loads one HF causal-LM + tokenizer and runs a single forward.

This is the "torch-op baseline" (Rule 1 in the project doc): the forward pass is
stock HuggingFace, so the engine stands up without any kernel work. Kernel
experiments (experiment 4) attach at `backends.py` and route individual ops
here — this class is the only place that touches the model, so that swap stays
localized.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import backends


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen3-8B"   # the doc's target; override for smoke tests
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    backend: str = "torch"            # see backends.py — only "torch" today


class ModelRunner:
    def __init__(self, config: ModelConfig):
        self.config = config
        backends.resolve(config.backend)  # validate up front; fail loud on typos

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.tokenizer.padding_side = "left"
        self.model = (
            AutoModelForCausalLM.from_pretrained(config.model_id, dtype=config.dtype)
            .to(config.device)
            .eval()
        )
        self.eos_token_ids = self._resolve_eos()

    def _resolve_eos(self) -> set[int]:
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        if eos is None:
            return set()
        return {eos} if isinstance(eos, int) else set(eos)

    def encode(self, text: list[str]) -> list[list[int]]:
        return self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.config.device)

    def decode(self, token_ids: list[torch.Tensor]) -> str:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask, cache):
        """One forward pass. `input_ids` is [batch, seq_len] — the full prompt on
        prefill, a single token on each decode step. Returns (logits, cache) where
        logits is [batch, seq_len, vocab] and `cache` is the updated KV cache.
        """
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
        return out.logits, out.past_key_values
