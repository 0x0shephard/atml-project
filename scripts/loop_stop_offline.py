"""Offline estimate of a loop-triggered stopping rule on saved chains (task T15, v2).

Rule: watch the thinking for a verbatim loop (the last ``window`` tokens already written
``repeats`` times in the last ``lookback`` tokens, checked every 256 tokens, as
crs.decoding.loops does live). When a loop is detected at token t, cut the thinking there and
force an answer. Chains that never loop either run free, or are cut at a cap C with a forced
answer ("+cap C").

The answer forced at t is not saved, so it is estimated gold-blind in one of two ways:
- **budget proxy** (8k runs, which have forced answers at 512/1k/2k/4k/8k): the forced answer at
  the largest budget B <= t. A loop between B and t adds only repeated text, so this should be
  close; it uses less of the chain, so it errs on the low side if the model was still progressing.
- **last candidate** (any run): the last \\boxed{} / "the answer is" value written before t;
  wrong if there is none. Used for the 32k runs, which have no forced answers.

Tokens: t + 30 for a loop stop, the forced record's tokens for a cap, the free run's otherwise.

Detectors are fixed in advance (64x8 is the live rule), and each loop stop is compared with the
same cap without it, so no choice is tuned on the results.

    python scripts/loop_stop_offline.py [--roots results,results-32k-ext]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crs.decoding.loops import LoopRule, first_loop_point  # noqa: E402
from crs.extract.answers import _ANSWER_IS, boxed_spans, is_equiv  # noqa: E402

BUDGETS = [512, 1024, 2048, 4096, 8192]
FORCE_TOKENS = 30
DETECTORS = {
    "64x8": LoopRule(64, 8, 8192),   # the live rule
    "32x4": LoopRule(32, 4, 4096),
    "32x3": LoopRule(32, 3, 4096),
    "16x3": LoopRule(16, 3, 2048),
}
SKIP = {"asvd-r80", "asvd-r85", "asvd-r90", "base-hf", "svd-r80", "svd-r95", "kv-int4"}


def load_ids(run: Path) -> dict[str, np.ndarray]:
    out = {}
    for f in sorted((run / "arrays").glob("*.npz")):
        z = np.load(f)
        for k in z.files:
            if k.endswith("/ids"):
                out[k[:-4]] = z[k]
    return out


def last_candidate(think: str, upto_frac: float) -> str | None:
    cut = int(len(think) * upto_frac)
    c = [(s, v) for s, _e, v in boxed_spans(think) if s < cut]
    c += [(m.start(1), m.group(1).strip()) for m in _ANSWER_IS.finditer(think) if m.start(1) < cut]
    return max(c)[1] if c else None


def per_chain(run: Path, dets: dict[str, LoopRule]) -> list[dict]:
    sc = [json.loads(l) for l in open(run / "scored.jsonl")]
    texts = {json.loads(l)["rid"]: json.loads(l).get("think_text", "") for l in open(run / "records.jsonl")}
    ids = load_ids(run)
    bud = {}
    for b in BUDGETS:
        f = run / f"scored_budget_{b}.jsonl"
        if f.exists():
            bud[b] = {r["rid"]: r for r in map(json.loads, open(f))}
    rows = []
    for r in sc:
        rid, nt = r["rid"], r["n_think_tokens"]
        row = {"pid": r["problem_id"], "free_ok": bool(r["correct"]), "free_tok": r["n_tokens"],
               "nt": nt, "forced": {}}
        for b, recs in bud.items():
            if rid in recs:
                row["forced"][b] = (bool(recs[rid]["correct"]), recs[rid]["n_tokens"] + (FORCE_TOKENS if recs[rid]["finish_reason"] == "forced" else 0))
        think, a = texts.get(rid, ""), ids.get(rid)
        row["loop"] = {}
        for name, rule in dets.items():
            t = first_loop_point(a[:nt], rule) if a is not None else None
            if t is None:
                row["loop"][name] = None
                continue
            bs = [b for b in BUDGETS if b <= t and b in row["forced"]]
            proxy = row["forced"][bs[-1]][0] if bs else (row["forced"][512][0] if 512 in row["forced"] else None)
            cand = last_candidate(think, t / max(nt, 1)) if think else None
            row["loop"][name] = {"t": t, "proxy_ok": proxy,
                                 "cand_ok": bool(cand and is_equiv(cand, r["gold"]))}
        rows.append(row)
    return rows


def evaluate(rows, det: str | None, cap: int | None, est: str) -> tuple[float, float, float] | None:
    """(right %, mean tokens, loop-fired %) for: loop stop with detector ``det`` (None = no loop
    stop) plus a forced answer at ``cap`` (None = run free)."""
    ok = tok = fired = 0.0
    for r in rows:
        lp = r["loop"].get(det) if det else None
        if cap is not None and cap not in r["forced"]:
            return None
        if lp is not None and (cap is None or lp["t"] < cap):
            v = lp["proxy_ok"] if est == "proxy" else lp["cand_ok"]
            if v is None:
                return None
            ok += v
            tok += lp["t"] + FORCE_TOKENS
            fired += 1
        elif cap is not None:
            ok += r["forced"][cap][0]
            tok += r["forced"][cap][1]
        else:
            ok += r["free_ok"]
            tok += r["free_tok"]
    n = len(rows)
    return 100 * ok / n, tok / n, 100 * fired / n


def fmt(x) -> str:
    return "–" if x is None else f"{x[0]:.1f} ({x[1] / 1000:.1f}K)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="results,results-32k-ext")
    ap.add_argument("--runs", default="*/*/none")
    a = ap.parse_args()

    all_rows = {}
    for root in a.roots.split(","):
        for scored in sorted(Path(root).glob(f"{a.runs}/scored.jsonl")):
            run = scored.parent
            if run.parts[-2] in SKIP:
                continue
            lim = "32k" if "32k" in root else "8k"
            all_rows[(lim, run.parts[-3], run.parts[-2])] = per_chain(run, DETECTORS)
            print(f"loaded {lim} {run.parts[-3]} {run.parts[-2]}", file=sys.stderr, flush=True)

    # 1. how often each detector fires, and on chains that were right in the free run
    print("## Loop detectors: share of chains flagged (flagged chains that were right in the free run)\n")
    print("| limit | model | setting | n | " + " | ".join(DETECTORS) + " |")
    print("|---|---|---|---|" + "---|" * len(DETECTORS))
    for (lim, m, s), rows in all_rows.items():
        cells = []
        for d in DETECTORS:
            f = [r for r in rows if r["loop"][d]]
            cells.append(f"{100 * len(f) / len(rows):.0f}% ({sum(r['free_ok'] for r in f)})")
        print(f"| {lim} | {m} | {s} | {len(rows)} | " + " | ".join(cells) + " |")

    # 2. does the budget proxy agree with the last-candidate estimate? (validates the 32k column)
    print("\n## Answer at the loop point: budget proxy vs last candidate (64x8 and 32x3 stops, 8k runs)\n")
    for d in ("64x8", "32x3"):
        agree = tot = pr = ca = 0
        for (lim, _m, _s), rows in all_rows.items():
            for r in rows:
                lp = r["loop"][d]
                if lp and lp["proxy_ok"] is not None:
                    tot += 1
                    agree += lp["proxy_ok"] == lp["cand_ok"]
                    pr += lp["proxy_ok"]
                    ca += lp["cand_ok"]
        print(f"- {d}: {tot} stops; right by proxy {pr}, by last candidate {ca}; agree on {agree}")

    # 3. same cap with and without the loop stop (no tuning: 64x8 is the live rule, fixed before
    #    this study; 32x3 is the loosest detector that rarely flags chains the free run got right)
    print("\n## Adding the loop stop, at the same cap\n")
    print("Cells: right % (mean tokens). Only chains with forced answers at every budget; the loop "
          "point's answer is the budget proxy. Δ = loop+cap minus cap alone, paired 95% bootstrap CI.\n")
    print("| model | setting | n | free run | loop 64x8 | loop 32x3 | 4k cap | 4k + 64x8 | 4k + 32x3 | Δ 4k+32x3 | 8k cap | 8k + 32x3 | Δ 8k+32x3 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    pooled = {4096: [], 8192: []}
    for (lim, m, s), rows in all_rows.items():
        rows = [r for r in rows if all(b in r["forced"] for b in BUDGETS)]
        if lim != "8k" or not rows:
            continue
        free = (100 * sum(r["free_ok"] for r in rows) / len(rows), sum(r["free_tok"] for r in rows) / len(rows), 0)
        cells = [fmt(free), fmt(evaluate(rows, "64x8", None, "proxy")), fmt(evaluate(rows, "32x3", None, "proxy"))]
        for cap in (4096, 8192):
            base = [chain_ok(r, None, cap) for r in rows]
            loop = [chain_ok(r, "32x3", cap) for r in rows]
            pooled[cap] += list(zip(base, loop))
            cells.append(fmt(evaluate(rows, None, cap, "proxy")))
            if cap == 4096:
                cells.append(fmt(evaluate(rows, "64x8", cap, "proxy")))
            cells.append(fmt(evaluate(rows, "32x3", cap, "proxy")))
            cells.append(ci(base, loop))
        print(f"| {m} | {s} | {len(rows)} | " + " | ".join(cells) + " |", flush=True)
    for cap, pairs in pooled.items():
        b, l = zip(*pairs)
        print(f"\nAll runs pooled, cap {cap}: Δ loop 32x3 = {ci(b, l)} over {len(b)} chains")

    # 4. 32k runs: loop stop alone, answer = last candidate before the loop (a lower bound)
    print("\n## 32k runs: loop stop alone (answer = last \\boxed / 'answer is' before the loop; lower bound)\n")
    print("| model | setting | n | free run | loop 64x8 | loop 32x3 |")
    print("|---|---|---|---|---|---|")
    for (lim, m, s), rows in all_rows.items():
        if lim != "32k":
            continue
        free = (100 * sum(r["free_ok"] for r in rows) / len(rows), sum(r["free_tok"] for r in rows) / len(rows), 0)
        print(f"| {m} | {s} | {len(rows)} | {fmt(free)} | {fmt(evaluate(rows, '64x8', None, 'cand'))} | "
              f"{fmt(evaluate(rows, '32x3', None, 'cand'))} |")


def chain_ok(r: dict, det: str | None, cap: int) -> bool:
    lp = r["loop"].get(det) if det else None
    if lp is not None and lp["t"] < cap:
        return bool(lp["proxy_ok"])
    return r["forced"][cap][0]


def ci(base, new, reps: int = 2000, seed: int = 0) -> str:
    d = np.asarray(new, float) - np.asarray(base, float)
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{100 * d.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"

if __name__ == "__main__":
    main()
