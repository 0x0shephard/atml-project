"""HF transformers generation backend.

Used for the KV-cache settings (which need a custom cache) and for local CPU tests.
Per-token statistics are recorded by a logits processor placed first in the chain, so it
sees the raw model logits, the same distribution vLLM reports logprobs for.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from ..compress.kvcache import FakeQuantKVCache
from ..decoding.penalties import EntropyPenalty
from ..record import GenResult, topk_entropy_torch
from .base import Backend, DecodeSpec


class StatsRecorder(LogitsProcessor):
    """Records the chosen-token log-prob and a top-k entropy estimate per step."""

    def __init__(self, k: int):
        self.k = k
        self.prev_logp: torch.Tensor | None = None
        self.lp: list[torch.Tensor] = []
        self.ent: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logp = torch.log_softmax(scores.float(), dim=-1)
        if self.prev_logp is not None:
            self.lp.append(self.prev_logp.gather(1, input_ids[:, -1:]).squeeze(1))
        self.ent.append(topk_entropy_torch(logp.topk(self.k, dim=-1).values))
        self.prev_logp = logp
        return scores

    def finish(self, last_tokens: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if self.prev_logp is not None:
            self.lp.append(self.prev_logp.gather(1, last_tokens[:, None]).squeeze(1))
        lp = torch.stack(self.lp, 1).cpu().numpy()
        ent = torch.stack(self.ent, 1).cpu().numpy()
        return lp, ent


class BiasProcessor(LogitsProcessor):
    def __init__(self, bias: dict[int, float]):
        self.ids = torch.tensor(list(bias.keys()), dtype=torch.long)
        self.vals = torch.tensor(list(bias.values()), dtype=torch.float32)

    def __call__(self, input_ids, scores):  # noqa: ANN001, ANN204
        ids = self.ids.to(scores.device)
        scores[:, ids] = scores[:, ids] + self.vals.to(scores.device, scores.dtype)
        return scores


class EntropyPenaltyProcessor(LogitsProcessor):
    def __init__(self, cfg: dict, prompt_len: int):
        self.pen = EntropyPenalty(**cfg)
        self.prompt_len = prompt_len
        self.closed: torch.Tensor | None = None

    def __call__(self, input_ids, scores):  # noqa: ANN001, ANN204
        if self.closed is None:
            self.closed = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=scores.device)
        if input_ids.shape[1] > self.prompt_len:
            self.closed |= input_ids[:, -1] == self.pen.think_end_last
        return self.pen.apply_batch(scores, ~self.closed)


class HFBackend(Backend):
    def __init__(
        self,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        dtype: str = "float16",
        kv: dict | None = None,
        batch_size: int = 16,
        model=None,  # noqa: ANN001 - an already-built model (tests, in-memory compression)
        tokenizer=None,  # noqa: ANN001
        device: str | None = None,
        **_: object,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        self.model = model.to(self.device).eval()
        self.tok = tokenizer or AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        self.kv = kv
        self.batch_size = batch_size
        gc = self.model.generation_config
        eos = gc.eos_token_id if gc.eos_token_id is not None else self.tok.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_ids[0]

    @torch.inference_mode()
    def _generate_batch(self, prompts: list[list[int]], spec: DecodeSpec, seed: int | None) -> list[GenResult]:
        L = max(len(p) for p in prompts)
        input_ids = torch.full((len(prompts), L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(prompts), L), dtype=torch.long)
        for i, p in enumerate(prompts):  # left padding
            input_ids[i, L - len(p):] = torch.tensor(p)
            attn[i, L - len(p):] = 1
        input_ids, attn = input_ids.to(self.device), attn.to(self.device)

        procs = LogitsProcessorList()
        rec = StatsRecorder(spec.logprobs_k) if spec.logprobs_k else None
        if rec:
            procs.append(rec)
        if spec.logit_bias:
            procs.append(BiasProcessor(spec.logit_bias))
        if spec.entropy_penalty:
            procs.append(EntropyPenaltyProcessor(spec.entropy_penalty, L))

        if seed is not None:
            torch.manual_seed(seed)
        kwargs = dict(
            max_new_tokens=spec.max_tokens,
            logits_processor=procs,
            pad_token_id=self.pad_id,
            eos_token_id=self.eos_ids,
        )
        if spec.greedy:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            kwargs.update(do_sample=True, temperature=spec.temperature, top_p=spec.top_p,
                          top_k=spec.top_k if spec.top_k and spec.top_k > 0 else 0)
        if self.kv:
            kwargs["past_key_values"] = FakeQuantKVCache(**self.kv)
        out = self.model.generate(input_ids=input_ids, attention_mask=attn, **kwargs)
        new = out[:, L:]
        lp = ent = None
        if rec is not None and new.shape[1] > 0:
            lp, ent = rec.finish(new[:, -1])
        results = []
        eos = set(self.eos_ids)
        for i in range(new.shape[0]):
            ids = new[i].tolist()
            end = next((j for j, t in enumerate(ids) if t in eos), None)
            finish = "stop" if end is not None else "length"
            n = end + 1 if end is not None else len(ids)
            ids = ids[:n]
            results.append(GenResult(
                ids,
                lp[i, :n] if lp is not None else None,
                ent[i, :n] if ent is not None else None,
                finish,
            ))
        return results

    def generate(self, prompt_ids: list[list[int]], spec: DecodeSpec) -> list[list[GenResult]]:
        # expand n samples per prompt, then batch; batches sorted by length to cut padding
        flat = [(i, s) for i in range(len(prompt_ids)) for s in range(spec.n)]
        order = sorted(range(len(flat)), key=lambda j: len(prompt_ids[flat[j][0]]))
        out: dict[tuple[int, int], GenResult] = {}
        for b in range(0, len(order), self.batch_size):
            idx = order[b : b + self.batch_size]
            batch = [prompt_ids[flat[j][0]] for j in idx]
            seed = None
            if spec.seeds:
                seed = spec.seeds[flat[idx[0]][0]] + 7919 * flat[idx[0]][1]
            for j, r in zip(idx, self._generate_batch(batch, spec, seed)):
                out[flat[j]] = r
        return [[out[(i, s)] for s in range(spec.n)] for i in range(len(prompt_ids))]

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
