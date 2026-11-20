"""Paired comparisons on the same problems (RQ1 per setting, RQ3 fixes)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .load import main_view, reference_setting

N_BOOT = 2000


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact McNemar test on paired booleans."""
    b01 = int(np.sum(~a & b))
    b10 = int(np.sum(a & ~b))
    if b01 + b10 == 0:
        return 1.0
    return float(stats.binomtest(b01, b01 + b10, 0.5).pvalue)


def boot_ci(diff: np.ndarray, rng: np.random.Generator, n: int = N_BOOT) -> tuple[float, float]:
    if len(diff) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue)


def compare(cur: pd.DataFrame, ref: pd.DataFrame, rng: np.random.Generator) -> dict:
    m = cur.merge(ref, on="problem_id", suffixes=("", "_ref"))
    if m.empty:
        return {"n": 0}
    c, r = m["correct"].to_numpy(bool), m["correct_ref"].to_numpy(bool)
    cf, cfr = m["commit_fail"].to_numpy(bool), m["commit_fail_ref"].to_numpy(bool)
    lt, ltr = m["n_think_tokens"].to_numpy(float), m["n_think_tokens_ref"].to_numpy(float)
    h, hr = m["hesitation_per_1k"].to_numpy(float), m["hesitation_per_1k_ref"].to_numpy(float)
    err = ~c
    lo, hi = boot_ci(c.astype(float) - r.astype(float), rng)
    clo, chi = boot_ci(cf.astype(float) - cfr.astype(float), rng)
    return {
        "n": len(m),
        "acc": c.mean(), "acc_ref": r.mean(), "d_acc": c.mean() - r.mean(), "d_acc_lo": lo, "d_acc_hi": hi,
        "p_acc_mcnemar": mcnemar_p(r, c),
        "commit_fail": cf.mean(), "commit_fail_ref": cfr.mean(), "d_commit_fail": cf.mean() - cfr.mean(),
        "d_commit_fail_lo": clo, "d_commit_fail_hi": chi, "p_commit_fail_mcnemar": mcnemar_p(cfr, cf),
        "commit_share_of_errors": cf[err].mean() if err.any() else np.nan,
        "never_correct": m["never_correct"].mean(),
        "unanswered": 1 - m["answered"].mean(),
        "think_tokens_mean": lt.mean(), "think_tokens_ref_mean": ltr.mean(),
        "length_ratio": lt.mean() / max(ltr.mean(), 1e-9), "p_length_wilcoxon": wilcoxon_p(lt, ltr),
        "hes_per_1k": h.mean(), "hes_per_1k_ref": hr.mean(), "p_hes_wilcoxon": wilcoxon_p(h, hr),
    }


def table(df: pd.DataFrame, ref_key: Callable[[str, str], tuple[str, str]], seed: int = 0) -> pd.DataFrame:
    """One row per (model, setting, fix) of the main view, compared with ``ref_key(setting, fix)``."""
    rng = np.random.default_rng(seed)
    mv = main_view(df)
    mv = mv[mv["samples"] == 1]
    rows = []
    for (model, setting, fix), cur in mv.groupby(["model", "setting", "fix"]):
        rs, rf = ref_key(setting, fix)
        if (rs, rf) == (setting, fix):
            continue
        ref = mv[(mv["model"] == model) & (mv["setting"] == rs) & (mv["fix"] == rf)]
        if ref.empty:
            continue
        cols = ["problem_id", "correct", "commit_fail", "never_correct", "answered", "n_think_tokens", "hesitation_per_1k"]
        res = compare(cur[cols], ref[cols], rng)
        rows.append({"model": model, "setting": setting, "fix": fix, "ref_setting": rs, "ref_fix": rf,
                     "family": cur["family"].iloc[0], "severity": cur["severity"].iloc[0], **res})
    return pd.DataFrame(rows)


def vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Each compressed version vs the uncompressed model, no fix (RQ1 per setting)."""
    t = table(df[df["fix"] == "none"], lambda s, f: (reference_setting(s), "none"))
    return t


def fix_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Each fix vs no fix on the same version (RQ3/RQ4/RQ5; on `base` it is the do-no-harm check)."""
    return table(df, lambda s, f: (s, "none"))
