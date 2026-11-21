"""vLLM generation backend (used for every weight-space setting).

Requires a CUDA GPU with compute capability >= 7.0 (Kaggle T4 is 7.5; P100 is not
supported). Logprobs are taken from the raw model distribution (vLLM V1 default
``logprobs_mode="raw_logprobs"``), before temperature, top-p and logit_bias.
"""

from __future__ import annotations

import numpy as np

from ..record import GenResult, topk_entropy
from .base import Backend, DecodeSpec


class VLLMBackend(Backend):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str | None = None,
        dtype: str = "float16",
        max_model_len: int = 9216,
        seed: int = 0,
        gpu_memory_utilization: float = 0.90,
        enable_entropy_penalty: bool = False,
        max_num_seqs: int = 256,
        attention_backend: str | None = None,
        **_: object,
    ):
        import os

        # T4 (compute 7.5) has no FlashAttention; vLLM 0.11 then picks FlexAttention, which
        # cannot use full CUDA graphs. Set e.g. TRITON_ATTN or XFORMERS here (see configs/bench.yaml).
        if attention_backend and attention_backend != "auto":
            os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
        from vllm import LLM

        kwargs: dict = {}
        if enable_entropy_penalty:
            from ..decoding.vllm_lp import EntropyPenaltyLogitsProcessor

            kwargs["logits_processors"] = [EntropyPenaltyLogitsProcessor]
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path or model_path,
            dtype=dtype,
            max_model_len=max_model_len,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_logprobs=64,
            **kwargs,
        )

    def _params(self, spec: DecodeSpec, seed: int | None):  # noqa: ANN202
        from vllm import SamplingParams

        extra = dict(spec.extra)
        if spec.entropy_penalty:
            extra["crs_entropy_penalty"] = spec.entropy_penalty
        return SamplingParams(
            n=spec.n,
            temperature=0.0 if spec.greedy else spec.temperature,
            top_p=1.0 if spec.greedy else spec.top_p,
            top_k=-1 if spec.greedy or spec.top_k is None or spec.top_k <= 0 else spec.top_k,
            max_tokens=spec.max_tokens,
            seed=seed,
            logprobs=spec.logprobs_k or None,
            logit_bias=spec.logit_bias or None,
            ignore_eos=spec.ignore_eos,
            detokenize=False,  # we decode ourselves; saves CPU on 20 logprobs/token
            extra_args=extra or None,
        )

    def generate(self, prompt_ids: list[list[int]], spec: DecodeSpec) -> list[list[GenResult]]:
        from vllm.inputs import TokensPrompt

        seeds = spec.seeds or [None] * len(prompt_ids)
        params = [self._params(spec, s) for s in seeds]
        prompts = [TokensPrompt(prompt_token_ids=p) for p in prompt_ids]
        outs = self.llm.generate(prompts, params, use_tqdm=True)
        k = spec.logprobs_k
        results: list[list[GenResult]] = []
        for o in outs:
            samples = []
            for c in sorted(o.outputs, key=lambda c: c.index):
                ids = list(c.token_ids)
                lp = ent = None
                if k and c.logprobs:
                    lp = np.empty(len(ids), dtype=np.float32)
                    ent = np.empty(len(ids), dtype=np.float32)
                    for i, (t, d) in enumerate(zip(ids, c.logprobs)):
                        lp[i] = d[t].logprob
                        ent[i] = topk_entropy([v.logprob for v in d.values() if v.rank is not None and v.rank <= k])
                samples.append(GenResult(ids, lp, ent, "stop" if c.finish_reason == "stop" else "length"))
            results.append(samples)
        return results

    def close(self) -> None:
        import gc

        import torch

        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
