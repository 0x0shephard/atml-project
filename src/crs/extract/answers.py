"""Final-answer extraction and equivalence checking."""

from __future__ import annotations

import re
from functools import lru_cache

_BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{")


def boxed_spans(text: str) -> list[tuple[int, int, str]]:
    """All ``\\boxed{...}`` contents with brace matching: (start, end, content)."""
    out = []
    for m in _BOXED.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append((m.start(), i, text[m.end():i - 1].strip()))
    return out


_ANSWER_IS = re.compile(
    r"(?:final answer|the answer|answer)\s*(?:is|should be|would be|=|:)\s*[:\s]*\$?\s*([^\n$]{1,60}?)\s*\$?(?:[.\n]|$)",
    re.IGNORECASE,
)


def final_answer(answer_text: str) -> str | None:
    """The model's committed answer: the last boxed answer after the thinking, falling
    back to an "answer is X" statement there. Unfinished thinking gives no answer."""
    spans = boxed_spans(answer_text)
    if spans:
        return spans[-1][2]
    m = list(_ANSWER_IS.finditer(answer_text))
    return m[-1].group(1).strip() if m else None


def _normalise(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right|,|!|;|:)", "", s)
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("^{\\circ}", ""), ("^\\circ", "")):
        s = s.replace(a, b)
    s = s.replace(" ", "").rstrip(".")
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


@lru_cache(maxsize=200_000)
def is_equiv(pred: str | None, gold: str) -> bool:
    if pred is None or pred.strip() == "":
        return False
    if _normalise(pred) == _normalise(gold):
        return True
    try:
        from math_verify import parse, verify

        g = parse(f"${gold}$")
        p = parse(f"${pred}$")
        if g and p:
            return bool(verify(g, p))
    except Exception:  # noqa: BLE001 - math_verify timeouts / parse errors count as no match
        pass
    try:
        return abs(float(_normalise(pred)) - float(_normalise(gold))) < 1e-6
    except ValueError:
        return False
