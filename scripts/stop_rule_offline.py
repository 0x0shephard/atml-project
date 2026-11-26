"""Offline estimate of an answer-stability stopping rule on saved chains (task T15).

Rule: read the thinking left to right and collect candidate answer statements without looking at
the gold answer: \\boxed{X}, "the answer is X", and sentences opening with So/Therefore/Thus/...
that end in "= X" / "is X" (values copied from the problem are ignored). Stop the thinking as soon
as the same value has been stated k times, and take that value as the answer. If the rule never
fires, the chain runs as it did (free run).

This assumes that forcing an answer at the stop point returns the repeated value, so it estimates
the live rule; a live run (truncate there + force the answer, as crs.run budget does) confirms it.
Tokens at the stop point are approximated from the character position.

    python scripts/stop_rule_offline.py [--k 2,3] [--roots results,results-32k-ext]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crs import data  # noqa: E402
from crs.extract.answers import _ANSWER_IS, _normalise, boxed_spans, is_equiv  # noqa: E402
from crs.extract.reasoning_scan import _CONCL_START, _SENT_SPLIT, _TAIL_VALUE, _in_text  # noqa: E402

FORCE_TOKENS = 30  # rough cost of the forced final answer


def blind_candidates(think: str, problem: str, strong_only: bool = False) -> list[tuple[int, str]]:
    out = [(s, c) for s, _e, c in boxed_spans(think)]
    out += [(m.start(1), m.group(1).strip()) for m in _ANSWER_IS.finditer(think)]
    if strong_only:
        return sorted(out)
    pos = 0
    for sent in _SENT_SPLIT.split(think):
        start = think.find(sent, pos)
        pos = start + len(sent) if start >= 0 else pos
        if start < 0 or not _CONCL_START.match(sent):
            continue
        m = _TAIL_VALUE.search(sent.rstrip())
        if m and not _in_text(m.group(1), problem):
            out.append((start + m.start(1), m.group(1).strip()))
    return sorted(out)


def _key(v: str) -> str:
    """Cheap identity for grouping repeated statements (the full is_equiv is kept for grading)."""
    n = _normalise(v)
    try:
        return repr(round(float(n), 9))
    except ValueError:
        return n


def stop_point(cands: list[tuple[int, str]], k: int) -> tuple[int, str] | None:
    counts: dict[str, int] = {}
    for ch, v in cands:
        v = v.strip().strip("$").rstrip(".")
        if not v or len(v) > 40:
            continue
        key = _key(v)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= k:
            return ch, v
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="2,3")
    ap.add_argument("--roots", default="results,results-32k-ext")
    ap.add_argument("--runs", default="*/*/none")
    ap.add_argument("--strong-only", action="store_true",
                    help="count only \\boxed{} and 'the answer is' statements, not So/Therefore sentences")
    a = ap.parse_args()
    ks = [int(x) for x in a.k.split(",")]
    probs = {p["id"]: p for p in data.load()}
    print("| limit | model | setting | n | free right % | free tokens | " +
          " | ".join(f"k={k} right % | k={k} tokens | k={k} fired %" for k in ks) + " | forced@4k % |")
    print("|---|---|---|---|---|---|" + "---|---|---|" * len(ks) + "---|")
    for root in a.roots.split(","):
        for scored in sorted(Path(root).glob(f"{a.runs}/scored.jsonl")):
            model, setting = scored.parts[-4], scored.parts[-3]
            if setting.startswith("svd") or setting in ("asvd-r80", "asvd-r85", "asvd-r90", "base-hf"):
                continue
            sc = {json.loads(l)["rid"]: json.loads(l) for l in open(scored)}
            texts = {}
            for l in open(scored.parent / "records.jsonl"):
                r = json.loads(l)
                texts[r["rid"]] = r.get("think_text", "")
            n = len(sc)
            free_ok = sum(r["correct"] for r in sc.values())
            free_tok = sum(r["n_tokens"] for r in sc.values()) / n
            cells = []
            for k in ks:
                ok, tok, fired = 0, 0.0, 0
                for rid, r in sc.items():
                    think = texts.get(rid, "")
                    sp = stop_point(blind_candidates(think, probs[r["problem_id"]]["problem"], a.strong_only), k) if think else None
                    if sp is None:
                        ok += r["correct"]
                        tok += r["n_tokens"]
                    else:
                        fired += 1
                        ok += is_equiv(sp[1], r["gold"])
                        tok += r["n_think_tokens"] * sp[0] / max(len(think), 1) + FORCE_TOKENS
                cells.append(f"{100 * ok / n:.1f} | {tok / n / 1000:.1f}K | {100 * fired / n:.0f}")
            fb = scored.parent / "scored_budget_4096.jsonl"
            f4 = "–"
            if fb.exists():
                b = [json.loads(l) for l in open(fb)]
                f4 = f"{100 * sum(x['correct'] for x in b) / len(b):.1f}" + ("" if len(b) == n else f" (n={len(b)})")
            lim = "32k" if "32k" in root else "8k"
            print(f"| {lim} | {model} | {setting} | {n} | {100 * free_ok / n:.1f} | {free_tok / 1000:.1f}K | "
                  + " | ".join(cells) + f" | {f4} |", flush=True)


if __name__ == "__main__":
    main()
