"""Print the results snapshot for LEDGER.md from the scored runs, with 95% intervals.

    python scripts/ledger_table.py            # markdown table of every scored run
    python scripts/ledger_table.py --vs-base  # adds the paired difference to base on shared problems

For each run folder <root>/<model>/<setting>/<fix>/scored.jsonl:
  n            problems scored
  right        share correct, with a Wilson 95% interval
  cut          share with no answer because the chain hit the limit (incl. loop stops)
  had-it       among wrong answers, the share whose reasoning contained the right answer
               (the "had it, then lost it" share Lotfi et al. report: 26% -> 52% under AWQ W3)
  forced@4k    accuracy when the thinking is cut at 4,096 tokens and an answer is forced
               (n shown when it covers fewer problems than the free run)
  vs base      (with --vs-base) accuracy minus base on the same problems, paired bootstrap 95% CI
Paste the output into the "Results snapshot" section of LEDGER.md.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOTS = ["results", "results-32k-ext"]


def load(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["rid"]] = r
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def paired_ci(diffs: list[int], iters: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (sum(diffs) / n, means[int(0.025 * iters)], means[int(0.975 * iters)])


def pct(x: float) -> str:
    return "–" if x != x else f"{100 * x:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vs-base", action="store_true")
    ap.add_argument("--fix", default="none")
    args = ap.parse_args()

    head = "| root | model | setting | n | right % (95% CI) | cut % | had-it % of errors | forced@4k % |"
    sep = "|---|---|---|---|---|---|---|---|"
    if args.vs_base:
        head += " vs base, pts (95% CI) |"
        sep += "---|"
    print(head)
    print(sep)
    for root in ROOTS:
        for scored in sorted(Path(root).glob(f"*/*/{args.fix}/scored.jsonl")):
            model, setting = scored.parts[-4], scored.parts[-3]
            recs = load(scored)
            n = len(recs)
            k = sum(r["correct"] for r in recs.values())
            lo, hi = wilson(k, n)
            cut = sum(r["outcome"] == "truncated" for r in recs.values()) / n
            wrong = [r for r in recs.values() if not r["correct"]]
            had = sum(bool(r.get("had_correct_scan")) for r in wrong) / len(wrong) if wrong else math.nan
            forced, forced_note = math.nan, ""
            fb = scored.parent / "scored_budget_4096.jsonl"
            if fb.exists():
                b = load(fb)
                forced = sum(r["correct"] for r in b.values()) / len(b) if b else math.nan
                if b and len(b) != n:
                    forced_note = f" (n={len(b)})"
            row = (f"| {root} | {model} | {setting} | {n} | {pct(k / n)} ({pct(lo)}–{pct(hi)}) | {pct(cut)} "
                   f"| {pct(had)} | {pct(forced)}{forced_note} |")
            if args.vs_base:
                base_path = Path(root) / model / "base" / args.fix / "scored.jsonl"
                if setting != "base" and base_path.exists():
                    base = load(base_path)
                    shared = sorted(set(recs) & set(base))
                    diffs = [int(recs[r]["correct"]) - int(base[r]["correct"]) for r in shared]
                    m, a, b_ = paired_ci(diffs)
                    row += f" {100 * m:+.1f} ({100 * a:+.1f} to {100 * b_:+.1f}), n={len(shared)} |"
                else:
                    row += " |"
            print(row)


if __name__ == "__main__":
    main()
