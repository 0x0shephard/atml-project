"""Load scored runs into one DataFrame."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .. import config
from ..record import iter_records

KEEP = ["rid", "problem_id", "source", "level", "gold", "model", "setting", "fix", "sample_idx", "finish_reason",
        "think_closed", "n_tokens", "n_think_tokens", "hesitation_count", "hesitation_per_1k", "mean_entropy_think",
        "mean_entropy_at_hesitation", "pred", "answered", "correct", "outcome", "outcome_strict", "outcome_judged",
        "had_correct_scan", "had_correct_strict", "had_correct_judged", "first_correct_token",
        "first_correct_token_judged", "first_correct_frac", "tokens_after_first_correct", "forced"]


def load_scored(results: Path, judged: bool = False) -> pd.DataFrame:
    settings = config.settings()
    rows = []
    for run in sorted(Path(results).glob("*/*/*")):
        if not (run / "scored.jsonl").exists():
            continue
        run_name = run.name  # e.g. "none" or "none-vote4"
        m = re.match(r"(.+)-vote(\d+)$", run_name)
        base_fix, samples = (m.group(1), int(m.group(2))) if m else (run_name, 1)
        files = [("scored.jsonl", None)] + [(f.name, int(f.stem.split("_")[-1])) for f in run.glob("scored_budget_*.jsonl")]
        for fname, budget in files:
            for r in iter_records(run, fname):
                row = {k: r.get(k) for k in KEEP}
                row.update(run=run_name, base_fix=base_fix, samples=samples, budget=budget)
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    st = df["setting"].map(lambda s: settings[s] if s in settings else None)
    df["family"] = st.map(lambda s: s.family if s else "?")
    df["severity"] = st.map(lambda s: s.severity if s else float("nan"))
    df["backend"] = st.map(lambda s: s.backend if s else "?")
    if judged and "outcome_judged" in df and df["outcome_judged"].notna().any():
        df["outcome"] = df["outcome_judged"].fillna(df["outcome"])
    df["commit_fail"] = df["outcome"] == "commit_fail"
    df["never_correct"] = df["outcome"] == "never_correct"
    df["truncated"] = df["outcome"] == "truncated"
    return df


def reference_setting(setting: str) -> str:
    """Paired reference for a setting: the uncompressed model on the same backend."""
    st = config.settings().get(setting)
    if st is None or setting in ("base", "base-hf"):
        return "base"
    if st.backend == "hf" and "base-hf" in config.settings():
        return "base-hf"
    return "base"


def main_view(df: pd.DataFrame) -> pd.DataFrame:
    """Full-budget, single-sample rows (sample 0 for voting runs)."""
    return df[df["budget"].isna() & (df["sample_idx"] == 0)]
