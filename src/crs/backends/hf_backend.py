"""HF transformers generation backend.

Used for the KV-cache settings (which need a custom cache) and for local CPU tests.

It runs its own decode loop instead of ``model.generate``. ``generate`` keeps a finished
sequence in the batch until the longest one ends, so every batch ran for the full 8,192
steps. In the 2026-11-21 main run that came to about 40 tok/s, roughly 12 h for one KV
setting. Here a finished sequence is dropped from the batch and the cache at once, so each
step only does work for the sequences still running.

Per-token statistics come from the raw model logits, before any bias, penalty or sampling
warper, which is the same distribution vLLM reports logprobs for. Sampling uses the HF
temperature / top-k / top-p warpers, in that order, as ``generate`` does.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import DynamicCache, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

from ..compress.kvcache import FakeQuantKVCache
from ..decoding.penalties import EntropyPenalty
from ..record import GenResult, topk_entropy_torch
from .base import Backend, DecodeSpec

# Prefill materialises [batch, prompt_len, intermediate] activations; cap the tokens per prefill so
# long prompts (the forced-answer step feeds whole 8k-token chains back in) run in smaller batches.
PREFILL_TOKEN_BUDGET = 16384


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
            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype)
        self.model = model.to(self.device).eval()
        self.tok = tokenizer or AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        self.kv = kv
        self.batch_size = batch_size
        gc = self.model.generation_config
        eos = gc.eos_token_id if gc.eos_token_id is not None else self.tok.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_ids[0]

    def _warpers(self, spec: DecodeSpec) -> list:
        if spec.greedy:
            return []
        w = []
        if spec.temperature and spec.temperature != 1.0:
            w.append(TemperatureLogitsWarper(spec.temperature))
        if spec.top_k and spec.top_k > 0:
            w.append(TopKLogitsWarper(spec.top_k))
        if spec.top_p is not None and spec.top_p < 1.0:
            w.append(TopPLogitsWarper(spec.top_p))
        return w

    @torch.inference_mode()
    def _generate_batch(self, prompts: list[list[int]], spec: DecodeSpec, seed: int | None) -> list[GenResult]:
        dev = self.device
        B, L, T = len(prompts), max(len(p) for p in prompts), spec.max_tokens
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for i, p in enumerate(prompts):  # left padding
            input_ids[i, L - len(p):] = torch.tensor(p)
            attn[i, L - len(p):] = 1
        input_ids, attn = input_ids.to(dev), attn.to(dev)

        gen = None
        if seed is not None and not spec.greedy:
            gen = torch.Generator(device=dev)
            gen.manual_seed(seed)
        warpers = self._warpers(spec)
        bias_ids = bias_vals = None
        if spec.logit_bias:
            bias_ids = torch.tensor(list(spec.logit_bias), dtype=torch.long, device=dev)
            bias_vals = torch.tensor(list(spec.logit_bias.values()), dtype=torch.float32, device=dev)
        pen = EntropyPenalty(**spec.entropy_penalty) if spec.entropy_penalty else None
        eos = torch.tensor(self.eos_ids, dtype=torch.long, device=dev)

        # outputs for every original row; `rows` maps the live batch back to them
        out_tok = torch.full((B, T), self.pad_id, dtype=torch.long, device=dev)
        out_lp = torch.zeros((B, T), dtype=torch.float32, device=dev)
        out_ent = torch.zeros((B, T), dtype=torch.float32, device=dev)
        length = torch.full((B,), T, dtype=torch.long, device=dev)
        stopped = torch.zeros(B, dtype=torch.bool, device=dev)
        rows = torch.arange(B, device=dev)
        thinking = torch.ones(B, dtype=torch.bool, device=dev)

        cache = FakeQuantKVCache(**self.kv) if self.kv else DynamicCache()
        pos = (attn.cumsum(-1) - 1).clamp_min(0)
        logits = self.model(input_ids=input_ids, attention_mask=attn, position_ids=pos, past_key_values=cache,
                            use_cache=True, logits_to_keep=1).logits[:, -1, :]

        for t in range(T):
            scores = logits.float()
            if spec.logprobs_k:
                logp = torch.log_softmax(scores, dim=-1)
                out_ent[rows, t] = topk_entropy_torch(logp.topk(spec.logprobs_k, dim=-1).values)
            if bias_ids is not None:
                scores[:, bias_ids] += bias_vals
            if pen is not None:
                scores = pen.apply_batch(scores, thinking)
            if spec.greedy:
                nxt = scores.argmax(-1)
            else:
                for w in warpers:
                    scores = w(None, scores)
                nxt = torch.multinomial(torch.softmax(scores, dim=-1), 1, generator=gen).squeeze(1)
            out_tok[rows, t] = nxt
            if spec.logprobs_k:
                out_lp[rows, t] = logp.gather(1, nxt[:, None]).squeeze(1)
            if pen is not None and pen.think_end_last is not None:
                thinking &= nxt != pen.think_end_last
            if t == T - 1:
                break
            done = torch.isin(nxt, eos) if not spec.ignore_eos else torch.zeros_like(nxt, dtype=torch.bool)
            if bool(done.any()):
                length[rows[done]] = t + 1
                stopped[rows[done]] = True
                keep = (~done).nonzero().squeeze(1)
                if keep.numel() == 0:
                    break
                cache.batch_select_indices(keep)
                rows, attn, nxt, thinking = rows[keep], attn[keep], nxt[keep], thinking[keep]
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=1)
            pos = attn.sum(-1, keepdim=True) - 1
            logits = self.model(input_ids=nxt[:, None], attention_mask=attn, position_ids=pos,
                                past_key_values=cache, use_cache=True).logits[:, -1, :]

        toks, lps, ents = out_tok.cpu().numpy(), out_lp.cpu().numpy(), out_ent.cpu().numpy()
        lens, stop = length.cpu().tolist(), stopped.cpu().tolist()
        del cache
        results = []
        for i in range(B):
            n = lens[i]
            results.append(GenResult(
                toks[i, :n].tolist(),
                lps[i, :n] if spec.logprobs_k else None,
                ents[i, :n] if spec.logprobs_k else None,
                "stop" if stop[i] else "length",
            ))
        return results

    def generate(self, prompt_ids: list[list[int]], spec: DecodeSpec) -> list[list[GenResult]]:
        # expand n samples per prompt, then batch; batches sorted by length to cut padding
        flat = [(i, s) for i in range(len(prompt_ids)) for s in range(spec.n)]
        order = sorted(range(len(flat)), key=lambda j: len(prompt_ids[flat[j][0]]))
        out: dict[tuple[int, int], GenResult] = {}
        b = 0
        while b < len(order):
            longest = len(prompt_ids[flat[order[min(b + self.batch_size, len(order)) - 1]][0]])
            size = max(1, min(self.batch_size, PREFILL_TOKEN_BUDGET // max(longest, 1)))
            idx = order[b : b + size]
            b += size
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
