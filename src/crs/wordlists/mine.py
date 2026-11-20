"""Automatically mined and random control word lists (RQ4).

``mine``: tokens over-represented at high-uncertainty points of compressed chains relative
to the baseline. Positions whose top-k entropy is above the baseline's ``quantile`` are
collected from both, and tokens are ranked by the log-odds ratio with an informative
Dirichlet prior (Monroe et al., 2008). Only word-start alphabetic tokens are kept.

``random_control``: for each hand-written word, a random word-start token of matching
frequency in baseline chains (same rank band), excluding every hesitation token.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

from ..record import iter_records, load_arrays


def _is_word_token(s: str) -> bool:
    w = s.strip()
    return len(w) >= 2 and w.isalpha() and w.isascii() and (s.startswith(" ") or w[0].isupper())


def _chains(runs: list[Path]):  # noqa: ANN202
    for run in runs:
        recs = list(iter_records(run))
        arrs = load_arrays(run, recs, ("ids", "ent"))
        for r in recs:
            a = arrs.get(r["rid"], {})
            if "ids" in a:
                n = r["n_think_tokens"]
                yield a["ids"][:n], (a["ent"][:n].astype(np.float32) if "ent" in a else None)


def high_entropy_counts(runs: list[Path], threshold: float) -> Counter:
    c: Counter = Counter()
    for ids, ent in _chains(runs):
        if ent is not None:
            c.update(ids[ent >= threshold].tolist())
    return c


def baseline_threshold(base_runs: list[Path], quantile: float) -> float:
    ents = [e for _ids, e in _chains(base_runs) if e is not None]
    return float(np.quantile(np.concatenate(ents), quantile))


def log_odds(c1: Counter, c2: Counter, a0: float = 1000.0) -> dict[int, float]:
    """z-scored log-odds of each token in c1 vs c2 with an informative Dirichlet prior."""
    tot = c1 + c2
    n1, n2, nt = sum(c1.values()), sum(c2.values()), sum(tot.values())
    out = {}
    for t, ct in tot.items():
        a = a0 * ct / nt
        y1, y2 = c1.get(t, 0), c2.get(t, 0)
        d = math.log((y1 + a) / (n1 + a0 - y1 - a)) - math.log((y2 + a) / (n2 + a0 - y2 - a))
        var = 1 / (y1 + a) + 1 / (y2 + a)
        out[t] = d / math.sqrt(var)
    return out


def mine(base_runs: list[Path], comp_runs: list[Path], tokenizer, k: int = 50, quantile: float = 0.9,  # noqa: ANN001
         min_count: int = 20) -> list[str]:
    thr = baseline_threshold(base_runs, quantile)
    cb = high_entropy_counts(base_runs, thr)
    cc = high_entropy_counts(comp_runs, thr)
    z = log_odds(cc, cb)
    words: list[str] = []
    for t, _score in sorted(z.items(), key=lambda kv: -kv[1]):
        if cc.get(t, 0) < min_count:
            continue
        s = tokenizer.decode([t])
        if _is_word_token(s) and s.strip() not in words:
            words.append(s.strip())
        if len(words) >= k:
            break
    return words


def random_control(hand_words: list[str], hesitation_ids: set[int], base_runs: list[Path], tokenizer,  # noqa: ANN001
                   seed: int = 0, band: float = 0.25) -> list[str]:
    freq: Counter = Counter()
    for ids, _ent in _chains(base_runs):
        freq.update(ids.tolist())
    ranked = [t for t, _ in freq.most_common()]
    rank = {t: i for i, t in enumerate(ranked)}
    pool = [t for t in ranked if t not in hesitation_ids and _is_word_token(tokenizer.decode([t]))]
    pool_rank = [rank[t] for t in pool]
    rng = random.Random(seed)
    chosen: list[str] = []
    used: set[str] = set()
    for w in hand_words:
        enc = tokenizer.encode(" " + w, add_special_tokens=False)
        r = rank.get(enc[0], len(ranked)) if len(enc) == 1 else len(ranked)
        lo, hi = r * (1 - band), max(r * (1 + band), r + 50)
        band_pool = [t for t, pr in zip(pool, pool_rank) if lo <= pr <= hi and tokenizer.decode([t]).strip() not in used]
        if not band_pool:
            band_pool = [t for t in pool[-500:] if tokenizer.decode([t]).strip() not in used]
        if not band_pool:
            continue
        s = tokenizer.decode([rng.choice(band_pool)]).strip()
        used.add(s)
        chosen.append(s)
    return chosen
