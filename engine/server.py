"""Engine: ties model + scheduler + sampler + KV cache together, plus a CLI.

`generate` batches all prompts into one padded forward: a single batched prefill,
then batched decode (one token per step for the whole batch) feeding the KV cache
forward until max_tokens. Left-padding + an attention mask keep the ragged prompt
lengths correct.
"""
from __future__ import annotations

import argparse

import torch

from .kv_cache import new_cache
from .model import ModelConfig, ModelRunner
from .sampler import SamplingParams, sample
from .scheduler import GenerationResult, Scheduler, Sequence


class Engine:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model = ModelRunner(self.config)
        self.scheduler = Scheduler()

    def generate(
        self,
        prompts: list[str],
        params: SamplingParams | None = None,
    ) -> list[GenerationResult]:
        params = params or [None]
        generated_tokens: list[torch.Tensor] = []

        seq = Sequence(self.model.encode(prompts), params or SamplingParams())
        self._run_to_completion(seq, generated_tokens)

        generated_tokens = torch.stack(generated_tokens, dim=1).squeeze(dim=-1)
        token_ids = generated_tokens
        decoded_tokens = self.model.decode(generated_tokens)

        return [
            self._result(prompt, output, token_id)
            for prompt, output, token_id in zip(prompts, decoded_tokens, token_ids)
        ]

    @torch.inference_mode()
    def _run_to_completion(self, seq: Sequence, generated_tokens: list[torch.Tensor]) -> None:
        device = self.config.device
        cache = new_cache()
        gen = _rng(seq.params, device)

        # Prefill: run the whole prompt; the last position predicts the 1st token.
        input_ids = seq.batch_encoding["input_ids"]
        attention_mask = seq.batch_encoding["attention_mask"]

        logits, cache = self.model.forward(input_ids, attention_mask, cache)
        next_logits = logits[:, -1, :]

        for _ in range(seq.params.max_tokens):
            token = sample(next_logits, seq.params, gen)
            generated_tokens.append(token)

            input_ids = torch.cat([token], dim=1)
            new_attention_mask = torch.ones((input_ids.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, new_attention_mask], dim=1)
            if token in self.model.eos_token_ids:
                seq.finish("stop")
                return
            # Decode: feed just the new token; the cache carries the context.
            logits, cache = self.model.forward(input_ids, attention_mask, cache)
            next_logits = logits[:, -1, :]
        seq.finish("length")

    def _result(self, prompt: str, output: str, output_token_ids: torch.Tensor) -> GenerationResult:
        return GenerationResult(
            prompt=prompt,
            text=prompt + output,
            output_token_ids=output_token_ids,
            finish_reason="length",
        )


def _rng(params: SamplingParams, device: str) -> torch.Generator | None:
    if params.greedy or params.seed is None:
        return None
    return torch.Generator().manual_seed(params.seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="nanoserve engine")
    ap.add_argument("--model", default=ModelConfig.model_id)
    ap.add_argument("--prompt", required=True, action="append",
                    help="prompt to generate from; repeat --prompt to pass several")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()  # action="append" makes args.prompt a list[str]

    # One shared sampling config for every prompt keeps the CLI simple; callers
    # needing per-prompt params can use the Engine.generate API directly.
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed)

    engine = Engine(ModelConfig(model_id=args.model))
    results = engine.generate(args.prompt, params)

    for i, r in enumerate(results):
        if len(results) > 1:
            print(f"=== prompt {i + 1}/{len(results)}: {r.prompt!r} ===")
        print(r.text)
        print(f"[{len(r.output_token_ids)} tokens, finish={r.finish_reason}]\n")
