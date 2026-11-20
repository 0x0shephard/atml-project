"""Build and freeze the 360-problem evaluation set.

200 MATH-500 problems (stratified by difficulty level), 100 GSM8K test problems and the
60 AIME 2024 + 2025 problems. The result is written once to ``data/problems.jsonl`` and
committed, so every model version solves exactly the same problems.

    python -m crs.data build            # writes data/problems.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import DATA_DIR

PROBLEMS_FILE = DATA_DIR / "problems.jsonl"

N_MATH = 200
N_GSM8K = 100
SEED = 20261108  # proposal date


def _first(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    raise KeyError(f"none of {list(keys)} in row with keys {list(row)}")


def _load_math500(rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    by_level: dict[int, list[dict]] = defaultdict(list)
    for i, r in enumerate(ds):
        by_level[int(r["level"])].append({
            "id": f"math500-{i:03d}",
            "source": "math500",
            "problem": r["problem"],
            "answer": r["answer"],
            "subject": r.get("subject"),
            "level": int(r["level"]),
        })
    # proportional allocation across levels, largest remainder
    total = sum(len(v) for v in by_level.values())
    quotas = {lv: N_MATH * len(v) / total for lv, v in by_level.items()}
    alloc = {lv: int(q) for lv, q in quotas.items()}
    for lv in sorted(quotas, key=lambda lv: quotas[lv] - alloc[lv], reverse=True)[: N_MATH - sum(alloc.values())]:
        alloc[lv] += 1
    out = []
    for lv in sorted(by_level):
        out.extend(rng.sample(by_level[lv], alloc[lv]))
    return out


def _gsm8k_answer(sol: str) -> str:
    ans = sol.split("####")[-1].strip()
    return ans.replace(",", "")


def _load_gsm8k(rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = rng.sample(range(len(ds)), N_GSM8K)
    return [{
        "id": f"gsm8k-{i:04d}",
        "source": "gsm8k",
        "problem": ds[i]["question"],
        "answer": _gsm8k_answer(ds[i]["answer"]),
        "subject": None,
        "level": None,
    } for i in sorted(idx)]


# (dataset id, config, split) candidates, tried in order.
AIME_SOURCES = {
    "aime24": [("HuggingFaceH4/aime_2024", None, "train"), ("Maxwell-Jia/AIME_2024", None, "train")],
    "aime25": [("math-ai/aime25", None, "test"), ("yentinglin/aime_2025", None, "train"),
               ("opencompass/AIME2025", "AIME2025-I", "test")],
}


def _load_aime(tag: str) -> list[dict]:
    from datasets import load_dataset

    last_err: Exception | None = None
    for name, cfg, split in AIME_SOURCES[tag]:
        try:
            ds = load_dataset(name, cfg, split=split)
            rows = list(ds)
            if name == "opencompass/AIME2025":  # two configs of 15 each
                rows += list(load_dataset(name, "AIME2025-II", split=split))
            out = []
            for i, r in enumerate(rows):
                ans = str(_first(r, ["answer", "Answer", "solution_answer"])).strip()
                ans = re.sub(r"^0+(?=\d)", "", ans)  # AIME answers are integers 0-999
                out.append({
                    "id": f"{tag}-{i:02d}",
                    "source": tag,
                    "problem": _first(r, ["problem", "Problem", "question"]),
                    "answer": ans,
                    "subject": None,
                    "level": None,
                })
            if len(out) != 30:
                raise ValueError(f"{name}: expected 30 problems, got {len(out)}")
            return out
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last_err = e
    raise RuntimeError(f"could not load {tag}: {last_err}")


def build(path: Path = PROBLEMS_FILE) -> list[dict]:
    rng = random.Random(SEED)
    problems = _load_math500(rng) + _load_gsm8k(rng) + _load_aime("aime24") + _load_aime("aime25")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    return problems


def load(path: Path = PROBLEMS_FILE, limit: int | None = None) -> list[dict]:
    with open(path) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    if limit is not None:
        # take a source-balanced head so small smoke runs cover every dataset
        by_src: dict[str, list[dict]] = defaultdict(list)
        for p in problems:
            by_src[p["source"]].append(p)
        out: list[dict] = []
        while len(out) < limit and any(by_src.values()):
            for src in list(by_src):
                if by_src[src] and len(out) < limit:
                    out.append(by_src[src].pop(0))
        problems = out
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "stats"])
    ap.add_argument("--out", type=Path, default=PROBLEMS_FILE)
    args = ap.parse_args()
    probs = build(args.out) if args.cmd == "build" else load(args.out)
    counts: dict[str, int] = defaultdict(int)
    for p in probs:
        counts[p["source"]] += 1
    print(f"{len(probs)} problems -> {args.out}: {dict(counts)}")


if __name__ == "__main__":
    main()
