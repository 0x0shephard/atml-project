r"""Detect a chain that is stuck repeating itself, so generation can stop it early.

Under heavy compression (3-bit RTN on R1-1.5B) many chains end in a verbatim loop, for example
"Therefore, the circle is the incircle for the two tangents." repeated until the length limit.
In the 32k run (2026-11-24) 26 of the 46 W3 chains still going at 32,768 tokens were such loops,
and each spent tens of thousands of tokens on them. A chain is called looping when its last
``window`` tokens occur at least ``repeats`` times, without overlapping, in its last ``lookback``
tokens: the same 64-token stretch has been written 8 times.

Checked on 2026-11-24 against all 3,988 saved R1 chains (8k and 32k runs, every setting), checking
every 256 tokens as generation does. The rule fires on 1,015 chains and would have saved 32% of
all tokens (46% of the W3 32k run). It fires on one chain scored correct, math500-046 under W3,
which had already written \boxed{6} before looping to the limit, so stopping it changes nothing.
Looser rules (4 repeats) stopped 7 chains that went on to answer correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoopRule:
    window: int = 64
    repeats: int = 8
    lookback: int = 8192
    every: int = 256  # check once per this many new tokens
    min_tokens: int = 1024  # never stop a chain shorter than this

    @classmethod
    def from_config(cls, cfg: dict | bool | None) -> LoopRule | None:
        if not cfg:
            return None
        return cls() if cfg is True else cls(**cfg)


def count_repeats(ids, window: int, lookback: int) -> int:  # noqa: ANN001
    """Non-overlapping occurrences of the last ``window`` tokens within the last ``lookback`` tokens."""
    a = np.asarray(ids[-lookback:], dtype=np.int64)
    n = len(a)
    if n < window:
        return 0
    tail = a[n - window:]
    starts = np.flatnonzero(a[: n - window + 1] == tail[0])
    count, last_end = 0, -1
    for s in starts[::-1]:  # from the end, so the tail itself counts and overlaps are skipped
        if s + window > last_end and last_end != -1:
            continue
        if np.array_equal(a[s:s + window], tail):
            count += 1
            last_end = s
    return count


def is_looping(ids, rule: LoopRule) -> bool:  # noqa: ANN001
    if len(ids) < max(rule.min_tokens, rule.window * rule.repeats):
        return False
    return count_repeats(ids, rule.window, rule.lookback) >= rule.repeats


def first_loop_point(ids, rule: LoopRule) -> int | None:  # noqa: ANN001
    """Token count at which an online check (every ``rule.every`` tokens) would stop this chain."""
    for n in range(rule.every, len(ids) + 1, rule.every):
        if is_looping(ids[:n], rule):
            return n
    return None
