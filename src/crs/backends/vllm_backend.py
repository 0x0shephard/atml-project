"""vLLM generation backend (used for every weight-space setting).

Requires a CUDA GPU with compute capability >= 7.0 (Kaggle T4 is 7.5; P100 is not
supported). Logprobs are taken from the raw model distribution (vLLM V1 default
``logprobs_mode="raw_logprobs"``), before temperature, top-p and logit_bias.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
import numpy as np

from ..record import GenResult, topk_entropy
from .base import Backend, DecodeSpec

log = logging.getLogger(__name__)


def _stats(ids: list[int], logprobs, k: int):  # noqa: ANN001, ANN202
    """Chosen-token logprob and top-k entropy per token, from vLLM's per-token logprob dicts."""
    if not (k and logprobs):
        return None, None
    lp = np.empty(len(ids), dtype=np.float32)
    ent = np.empty(len(ids), dtype=np.float32)
    for i, (t, d) in enumerate(zip(ids, logprobs)):
        lp[i] = d[t].logprob
        ent[i] = topk_entropy([v.logprob for v in d.values() if v.rank is not None and v.rank <= k])
    return lp, ent


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
                lp, ent = _stats(ids, c.logprobs, k)
                samples.append(GenResult(ids, lp, ent, "stop" if c.finish_reason == "stop" else "length"))
            results.append(samples)
        return results

    def generate_stream(self, prompt_ids: list[list[int]], spec: DecodeSpec,
                        on_done: Callable[[int, list[GenResult]], None],
                        stop_check: Callable[[list[int]], bool] | None = None, chunk: int = 20,
                        check_every: int = 256) -> None:
        """All prompts go to the engine at once, so a finished chain's slot is refilled at once
        instead of waiting for the slowest chain of a chunk. Each chain is handed to ``on_done``
        as soon as it ends; ``stop_check`` runs every ``check_every`` new tokens and aborts a
        chain it flags (finish_reason "loop")."""
        if spec.n != 1:  # vLLM fans n>1 out into child requests; keep the simple chunked path
            return super().generate_stream(prompt_ids, spec, on_done, stop_check, chunk)
        from vllm.inputs import TokensPrompt
        from vllm.sampling_params import RequestOutputKind

        eng = self.llm.llm_engine
        seeds = spec.seeds or [None] * len(prompt_ids)
        self._stream_no = getattr(self, "_stream_no", 0) + 1
        live: dict[str, dict] = {}
        for i, (p, sd) in enumerate(zip(prompt_ids, seeds)):
            rid = f"crs{self._stream_no}-{i}"
            # DELTA: each step returns only the new tokens, so a 32k chain is not re-sent every step
            params = self._params(spec, sd)
            params.output_kind = RequestOutputKind.DELTA  # a msgspec struct; vLLM's own LLM sets this the same way
            eng.add_request(rid, TokensPrompt(prompt_token_ids=p), params)
            live[rid] = {"i": i, "ids": [], "lp": [], "ent": [], "next": check_every}
        k = spec.logprobs_k
        n_done = n_loop = n_tok = 0
        t0 = t_log = time.time()

        def finish(rid: str, reason: str) -> None:
            st = live.pop(rid)
            lp = np.concatenate(st["lp"]) if st["lp"] else None
            ent = np.concatenate(st["ent"]) if st["ent"] else None
            on_done(st["i"], [GenResult(st["ids"], lp, ent, reason)])

        while eng.has_unfinished_requests():
            for out in eng.step():
                st = live.get(out.request_id)
                if st is None:  # output of a chain already stopped
                    continue
                c = out.outputs[0]
                new = list(c.token_ids)
                st["ids"].extend(new)
                n_tok += len(new)
                lp, ent = _stats(new, c.logprobs, k)
                if lp is not None:
                    st["lp"].append(lp)
                    st["ent"].append(ent)
                if out.finished:
                    n_done += 1
                    finish(out.request_id, "stop" if c.finish_reason == "stop" else "length")
                elif stop_check is not None and len(st["ids"]) >= st["next"]:
                    st["next"] = len(st["ids"]) + check_every
                    if stop_check(st["ids"]):
                        eng.abort_request([out.request_id])
                        n_done += 1
                        n_loop += 1
                        finish(out.request_id, "loop")
            if time.time() - t_log > 300:
                t_log = time.time()
                log.info("stream: %d/%d done (%d stopped looping), %.0f tok/s", n_done, len(prompt_ids), n_loop,
                         n_tok / (t_log - t0))
        for rid in list(live):  # should not happen; never drop a chain silently
            log.warning("stream: %s ended without a final output", rid)
            finish(rid, "length")
        log.info("stream: %d chains, %d stopped looping, %.0f tok/s", n_done, n_loop, n_tok / max(time.time() - t0, 1e-9))

    def close(self) -> None:
        import gc

        import torch

        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
