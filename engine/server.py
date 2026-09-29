"""Engine: ties model + scheduler + sampler + KV cache together, plus a CLI.

Single-request baseline: prompts are processed one at a time (no batching). Each
prompt runs its own prefill, then a one-token-per-step greedy decode that feeds
the KV cache forward until an EOS token or max_tokens.
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
        params = params or SamplingParams()
        order: list[Sequence] = []
        for prompt in prompts:
            seq = Sequence(self.model.encode(prompt), params)
            self.scheduler.add(seq)
            order.append(seq)

        # Baseline: drain FIFO, each sequence run to completion, one at a time.
        while self.scheduler.has_waiting():
            self._run_to_completion(self.scheduler.pop())

        return [self._result(prompt, seq) for prompt, seq in zip(prompts, order)]

    @torch.inference_mode()
    def _run_to_completion(self, seq: Sequence) -> None:
        device = self.config.device
        cache = new_cache()
        gen = _rng(seq.params, device)

        # Prefill: run the whole prompt; the last position predicts the 1st token.
        input_ids = torch.tensor([seq.prompt_ids])
        logits, cache = self.model.forward(input_ids, cache)
        next_logits = logits[0, -1]

        for _ in range(seq.params.max_tokens):
            token = sample(next_logits, seq.params, gen)
            seq.append(token)
            if token in self.model.eos_token_ids:
                seq.finish("stop")
                return
            # Decode: feed just the new token; the cache carries the context.
            logits, cache = self.model.forward(torch.tensor([[token]]), cache)
            next_logits = logits[0, -1]
        seq.finish("length")

    def _result(self, prompt: str, seq: Sequence) -> GenerationResult:
        return GenerationResult(
            prompt=prompt,
            text=prompt + self.model.decode(seq.output_ids),
            output_token_ids=seq.output_ids,
            finish_reason=seq.finish_reason or "length",
        )


def _rng(params: SamplingParams, device: str) -> torch.Generator | None:
    if params.greedy or params.seed is None:
        return None
    return torch.Generator().manual_seed(params.seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="mini-serve engine")
    ap.add_argument("--model", default=ModelConfig.model_id)
    ap.add_argument("--prompt", required=True, action="append",
                    help="prompt to generate from; repeat --prompt to pass several")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()  # action="append" makes args.prompt a list[str]

    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed)
    engine = Engine(ModelConfig(model_id=args.model))
    results = engine.generate(args.prompt, params)

    for i, r in enumerate(results):
        if len(results) > 1:
            print(f"=== prompt {i + 1}/{len(results)}: {r.prompt!r} ===")
        print(r.text)
        print(f"[{len(r.output_token_ids)} tokens, finish={r.finish_reason}]\n")
