"""Find where (if anywhere) the correct answer first appears inside the reasoning.

A simple rule collects candidate answer statements from the thinking text and checks each
against the gold answer. Candidates are tagged by how strong a commitment they are:

* ``boxed``      - ``\\boxed{X}`` inside the thinking
* ``answer_is``  - "the answer is X", "answer: X", ...
* ``conclusion`` - a sentence opening with So/Therefore/Thus/Hence/... that ends in
                   "= X" / "is X" (or, for numeric gold answers, contains it as a number)

Hits are confirmed by the judge model (``crs.extract.judge``) and a 10% manual check,
since an intermediate quantity can equal the final answer by coincidence.

Validation (2026-11-22, 306 hand-labelled spans from 117 failed chains, ``data/validation/``):
``boxed``/``answer_is`` hits were always real, but only 67% of ``conclusion`` hits were. A
third of the false ones repeated a value given in the problem ("the angle is 60 degrees"),
so a ``conclusion`` hit whose value appears in the problem statement is dropped when the
problem text is passed in. That removed 23 of 69 false conclusion spans and 1 of 141 real ones.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import asdict, dataclass

from .answers import _ANSWER_IS, boxed_spans, is_equiv

_SENT_SPLIT = re.compile(r"(?<=[.!?\n])\s+")
_CONCL_START = re.compile(r"^\s*(?:\*\*)?(so|therefore|thus|hence|that means|which means|this gives|so the|then the)\b", re.IGNORECASE)
_TAIL_VALUE = re.compile(r"(?:=|\bis\b|\bequals\b|\bget\b|\bgives\b)\s*\$*\s*([^=$\n]{1,40}?)\s*\$*\s*[.!]?\s*$", re.IGNORECASE)
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class Hit:
    char: int
    kind: str
    text: str
    token: int = -1


def _numeric(gold: str) -> float | None:
    g = gold.strip().replace(",", "").strip("$")
    try:
        return float(g)
    except ValueError:
        return None


def _in_text(value: str, text: str) -> bool:
    v = value.strip().strip("$").replace(",", "").strip()
    if not v:
        return False
    return re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", text.replace(",", "")) is not None


def candidates(think: str, gold: str, problem: str | None = None) -> list[Hit]:
    out: list[Hit] = []
    for s, _e, content in boxed_spans(think):
        out.append(Hit(s, "boxed", content))
    for m in _ANSWER_IS.finditer(think):
        out.append(Hit(m.start(1), "answer_is", m.group(1).strip()))
    gnum = _numeric(gold)
    pos = 0
    for sent in _SENT_SPLIT.split(think):
        start = think.find(sent, pos)
        pos = start + len(sent) if start >= 0 else pos
        if start < 0 or not _CONCL_START.match(sent):
            continue
        m = _TAIL_VALUE.search(sent.rstrip())
        if m:
            out.append(Hit(start + m.start(1), "conclusion", m.group(1).strip()))
        elif gnum is not None:
            for nm in _NUM.finditer(sent):
                try:
                    if abs(float(nm.group().replace(",", "")) - gnum) < 1e-9:
                        out.append(Hit(start + nm.start(), "conclusion", nm.group()))
                        break
                except ValueError:
                    continue
    if problem:
        out = [h for h in out if not (h.kind == "conclusion" and _in_text(h.text, problem))]
    out.sort(key=lambda h: h.char)
    return out


def char_to_token(think: str, tokenizer) -> callable:  # noqa: ANN001
    """Map character offsets in ``think`` to token indices (re-tokenisation offsets)."""
    enc = tokenizer(think, add_special_tokens=False, return_offsets_mapping=True)
    starts = [a for a, _b in enc["offset_mapping"]]
    return lambda c: max(0, bisect.bisect_right(starts, c) - 1)


def scan(think: str, gold: str, tokenizer=None, max_hits: int = 20, problem: str | None = None) -> dict:  # noqa: ANN001
    cands = candidates(think, gold, problem)
    correct = [h for h in cands if is_equiv(h.text, gold)]
    if tokenizer is not None and correct:
        c2t = char_to_token(think, tokenizer)
        for h in correct:
            h.token = c2t(h.char)
    first = correct[0] if correct else None
    strict = [h for h in correct if h.kind in ("boxed", "answer_is")]
    return {
        "n_candidates": len(cands),
        "n_correct_candidates": len(correct),
        "had_correct_scan": bool(correct),
        "had_correct_strict": bool(strict),
        "first_correct_char": first.char if first else None,
        "first_correct_token": first.token if first else None,
        "first_correct_kind": first.kind if first else None,
        "last_candidate_correct": bool(cands) and is_equiv(cands[-1].text, gold),
        "correct_hits": [asdict(h) for h in correct[:max_hits]],
    }
