"""Hesitation-word penalties.

Word lists are turned into token ids by taking every surface variant of each word
(``Wait``, `` Wait``, ``\\nWait`` ...) that the tokenizer encodes as exactly one token.
Words with no single-token variant are reported and skipped, so a multi-token word never
causes its (often innocent) first sub-token to be penalised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import WORDLIST_DIR

VARIANT_PREFIXES = ("", " ", "\n", "\n\n")


def read_wordlist(name_or_path: str | Path, model_key: str | None = None) -> list[str]:
    """Load a word list: a path, or a name in ``crs/wordlists`` (``handwritten``,
    ``mined``/``random`` resolve to ``<name>_<model>.txt``)."""
    p = Path(name_or_path)
    if not p.exists():
        candidates = [WORDLIST_DIR / f"{name_or_path}.txt"]
        if model_key:
            candidates.insert(0, WORDLIST_DIR / f"{name_or_path}_{model_key}.txt")
        p = next((c for c in candidates if c.exists()), candidates[-1])
    if not p.exists():
        raise FileNotFoundError(f"word list {name_or_path!r} not found (looked for {p}); "
                                f"run `python -m crs.run wordlists` to create mined/random lists")
    words = []
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].rstrip("\n")
        if line.strip():
            words.append(line.strip())
    return words


@dataclass
class ExpandedList:
    token_ids: list[int]
    variants: dict[str, list[int]]  # word -> token ids of its single-token variants
    skipped: list[str]


def expand_wordlist(words: list[str], tokenizer) -> ExpandedList:  # noqa: ANN001
    ids: set[int] = set()
    variants: dict[str, list[int]] = {}
    skipped = []
    for w in words:
        found = []
        for pre in VARIANT_PREFIXES:
            enc = tokenizer.encode(pre + w, add_special_tokens=False)
            if len(enc) == 1:
                found.append(enc[0])
        if found:
            variants[w] = sorted(set(found))
            ids.update(found)
        else:
            skipped.append(w)
    return ExpandedList(sorted(ids), variants, skipped)


def logit_bias(token_ids: list[int], alpha: float) -> dict[int, float]:
    return {int(t): -float(alpha) for t in token_ids}


def save_expanded(path: Path, exp: ExpandedList) -> None:
    path.write_text(json.dumps({"token_ids": exp.token_ids, "variants": exp.variants, "skipped": exp.skipped}, indent=1))


class EntropyPenalty:
    """Per-request callable: penalty = alpha_max * sigmoid((H - h0) / tau) on the listed
    tokens, where H is the entropy (nats) of the current next-token distribution.

    Applied only while the model is still thinking (before the think-end token).
    Works on a 1-D logits row (vLLM per-request processors) or a 2-D batch (HF).
    """

    def __init__(self, token_ids: list[int], alpha_max: float, h0: float, tau: float,
                 think_end_ids: list[int] | None = None):
        self.token_ids = list(token_ids)
        self.alpha_max = float(alpha_max)
        self.h0 = float(h0)
        self.tau = float(tau)
        self.think_end_last = (think_end_ids or [None])[-1]
        self._idx = None
        self.closed = False  # 1-D (single request) state

    def penalty(self, logits):  # noqa: ANN001, ANN201
        import torch

        logp = torch.log_softmax(logits.float(), dim=-1)
        h = -(logp.exp() * logp).sum(-1)
        return self.alpha_max * torch.sigmoid((h - self.h0) / self.tau)

    def _ids(self, device):  # noqa: ANN001, ANN202
        import torch

        if self._idx is None or self._idx.device != device:
            self._idx = torch.tensor(self.token_ids, dtype=torch.long, device=device)
        return self._idx

    def __call__(self, output_ids: list[int], logits):  # noqa: ANN001, ANN204
        """vLLM per-request signature: (generated ids so far, 1-D logits) -> logits."""
        if not self.closed and output_ids and output_ids[-1] == self.think_end_last:
            self.closed = True
        if self.closed:
            return logits
        pen = self.penalty(logits)
        idx = self._ids(logits.device)
        logits[idx] = logits[idx] - pen.to(logits.dtype)
        return logits

    def apply_batch(self, logits, active):  # noqa: ANN001, ANN201
        """HF: 2-D logits [B, V]; ``active`` is a bool tensor [B] (still thinking)."""
        pen = self.penalty(logits) * active.to(logits.dtype)
        idx = self._ids(logits.device)
        logits[:, idx] = logits[:, idx] - pen[:, None].to(logits.dtype)
        return logits
