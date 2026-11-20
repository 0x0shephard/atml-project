"""Severity trends (RQ2), pruning length / budget / voting checks (RQ6)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from ..extract.answers import is_equiv
from .load import main_view, reference_setting


def severity_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of severity with per-record metrics, reference included at 0."""
    mv = main_view(df)
    mv = mv[(mv["fix"] == "none") & (mv["samples"] == 1)]
    rows = []
    for (model, family), fam in mv.groupby(["model", "family"]):
        if family.startswith("none"):
            continue
        ref = mv[(mv["model"] == model) & (mv["setting"] == reference_setting(fam["setting"].iloc[0]))].copy()
        ref["severity"] = 0.0
        d = pd.concat([ref, fam])
        for metric in ["hesitation_per_1k", "n_think_tokens", "commit_fail", "correct", "mean_entropy_at_hesitation"]:
            x = d[["severity", metric]].dropna()
            if x["severity"].nunique() < 2 or len(x) < 10:
                continue
            rho, p = stats.spearmanr(x["severity"], x[metric].astype(float))
            per_level = x.groupby("severity")[metric].mean().to_dict()
            rows.append({"model": model, "family": family, "metric": metric, "spearman_rho": rho, "p": p,
                         "n_levels": x["severity"].nunique(), "means_by_severity": per_level})
    return pd.DataFrame(rows)


def budget_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and commit-fail rate per thinking budget (full budget = max_tokens)."""
    d = df[(df["fix"].str.split("@").str[0] == "none") & (df["samples"] == 1) & (df["sample_idx"] == 0)].copy()
    d["budget"] = d["budget"].fillna(np.inf)
    g = d.groupby(["model", "setting", "family", "severity", "budget"])
    return g.agg(acc=("correct", "mean"), commit_fail=("commit_fail", "mean"),
                 think_tokens=("n_think_tokens", "mean"), n=("correct", "size")).reset_index()


def pruning_gain_vs_budget(curves: pd.DataFrame) -> pd.DataFrame:
    """Accuracy gain of each pruned setting over the reference at every budget (RQ6):
    if light pruning helps only by stopping earlier, the gain should shrink as budget grows."""
    rows = []
    for (model, budget), g in curves.groupby(["model", "budget"]):
        ref = g[g["setting"] == "base"]
        if ref.empty:
            continue
        for _, r in g[g["family"].isin(["wanda", "magnitude"])].iterrows():
            rows.append({"model": model, "budget": budget, "setting": r["setting"], "severity": r["severity"],
                         "gain": r["acc"] - ref["acc"].iloc[0], "think_tokens": r["think_tokens"],
                         "think_tokens_ref": ref["think_tokens"].iloc[0]})
    return pd.DataFrame(rows)


def majority_vote(preds: list[str | None]) -> str | None:
    """Majority answer with mathematical-equivalence clustering; ties go to the earliest."""
    clusters: list[list[str]] = []
    for p in preds:
        if p is None:
            continue
        for c in clusters:
            if is_equiv(p, c[0]):
                c.append(p)
                break
        else:
            clusters.append([p])
    if not clusters:
        return None
    return max(clusters, key=len)[0]


def voting_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["budget"].isna() & (df["samples"] > 1)]
    rows = []
    for (model, setting, run), g in d.groupby(["model", "setting", "run"]):
        accs, single = [], []
        for pid, gp in g.groupby("problem_id"):
            gp = gp.sort_values("sample_idx")
            vote = majority_vote(list(gp["pred"]))
            accs.append(is_equiv(vote, gp["gold"].iloc[0]))
            single.append(gp["correct"].mean())
        rows.append({"model": model, "setting": setting, "run": run, "n": len(accs),
                     "vote_acc": float(np.mean(accs)), "mean_single_acc": float(np.mean(single)),
                     "think_tokens": g["n_think_tokens"].mean()})
    return pd.DataFrame(rows)


def answer_agreement(preds: list[str | None]) -> float:
    c = Counter(p for p in preds if p is not None)
    return max(c.values()) / len(preds) if c else 0.0
