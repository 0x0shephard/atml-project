"""10% manual check of the answer-finding rule and the judge.

``export`` draws a stratified sample (by family x judge verdict) of judged hits plus a
share of "no hit" failures, and writes a CSV with an empty ``human_label`` column
(YES = the solver states the gold answer as its answer at that span; for no-hit rows,
YES = the correct answer does appear somewhere in the reasoning).
``agreement`` reads the labelled CSV back and reports judge/scan precision and Cohen's kappa.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from ..record import iter_records
from .judge import excerpt

FIELDS = ["run", "rid", "family", "kind", "scan_kind", "judge", "gold", "problem", "excerpt", "human_label"]


def export(runs: list[Path], problems: dict[str, dict], settings: dict, out: Path, frac: float = 0.10, seed: int = 0) -> int:
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = {}
    for run in runs:
        texts = {r["rid"]: r["think_text"] for r in iter_records(run)}
        for row in iter_records(run, "scored.jsonl"):
            fam = settings[row["setting"]].family if row["setting"] in settings else "?"
            hits = row.get("correct_hits", [])
            verdicts = row.get("judge_verdicts") or []
            if hits:
                h = hits[0]
                item = dict(run=str(run), rid=row["rid"], family=fam, kind="hit", scan_kind=h["kind"],
                            judge=verdicts[0] if verdicts else None, gold=row["gold"],
                            problem=problems[row["problem_id"]]["problem"],
                            excerpt="...{}<<<{}>>>{}...".format(*excerpt(texts[row["rid"]], h["char"], h["text"])),
                            human_label="")
                strata.setdefault((fam, "hit", item["judge"]), []).append(item)
            elif not row["correct"]:
                item = dict(run=str(run), rid=row["rid"], family=fam, kind="no_hit", scan_kind="", judge=None,
                            gold=row["gold"], problem=problems[row["problem_id"]]["problem"],
                            excerpt=texts[row["rid"]][-3000:], human_label="")
                strata.setdefault((fam, "no_hit", None), []).append(item)
    sample = []
    for items in strata.values():
        k = max(1, round(frac * len(items)))
        sample.extend(rng.sample(items, min(k, len(items))))
    rng.shuffle(sample)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sample)
    return len(sample)


def agreement(path: Path) -> dict:
    rows = [r for r in csv.DictReader(open(path)) if r["human_label"].strip()]
    hits = [r for r in rows if r["kind"] == "hit" and r["judge"] in ("True", "False")]
    human = [r["human_label"].strip().upper().startswith("Y") for r in hits]
    judge = [r["judge"] == "True" for r in hits]
    n = len(hits)
    out = {"n_labelled": len(rows), "n_hits": n}
    if n:
        po = sum(a == b for a, b in zip(human, judge)) / n
        ph, pj = sum(human) / n, sum(judge) / n
        pe = ph * pj + (1 - ph) * (1 - pj)
        out.update({
            "scan_precision": sum(human) / n,  # how often a raw scan hit is real
            "judge_accuracy": po,
            "judge_precision": (sum(h and j for h, j in zip(human, judge)) / max(sum(judge), 1)),
            "judge_recall": (sum(h and j for h, j in zip(human, judge)) / max(sum(human), 1)),
            "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else 1.0,
        })
    nohit = [r for r in rows if r["kind"] == "no_hit"]
    if nohit:
        out["no_hit_miss_rate"] = sum(r["human_label"].strip().upper().startswith("Y") for r in nohit) / len(nohit)
    return out
