"""Compare compression methods at equal accuracy loss (RQ1).

For each model and family, the per-setting accuracy drop (vs the paired uncompressed
reference) and a failure metric form a curve through (0, reference value). The metric is
linearly interpolated at target drops (5/10/15 points); a target outside a family's
observed range gives NaN rather than an extrapolation. Confidence intervals come from a
problem-level bootstrap that recomputes every point of the curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .load import main_view, reference_setting

METRICS = ["commit_fail", "commit_share_of_errors", "never_correct", "hesitation_per_1k", "n_think_tokens"]
TARGETS = (0.05, 0.10, 0.15)


def _point(cur: pd.DataFrame, metric: str) -> float:
    if metric == "commit_share_of_errors":
        err = ~cur["correct"]
        return float(cur.loc[err, "commit_fail"].mean()) if err.any() else np.nan
    return float(cur[metric].astype(float).mean())


def _curve(frames: list[tuple[float, pd.DataFrame]], ref: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    acc_ref = ref["correct"].mean()
    drops = [0.0] + [acc_ref - f["correct"].mean() for _s, f in frames]
    vals = [_point(ref, metric)] + [_point(f, metric) for _s, f in frames]
    order = np.argsort(drops, kind="stable")
    return np.asarray(drops)[order], np.asarray(vals)[order]


def _interp(drops: np.ndarray, vals: np.ndarray, t: float) -> float:
    ok = np.isfinite(vals)
    d, v = drops[ok], vals[ok]
    if len(d) < 2 or t < d.min() or t > d.max():
        return np.nan
    return float(np.interp(t, d, v))


def equal_loss_table(df: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mv = main_view(df)
    mv = mv[(mv["fix"] == "none") & (mv["samples"] == 1)]
    rows = []
    for (model, family), fam in mv.groupby(["model", "family"]):
        if family.startswith("none"):
            continue
        ref_name = reference_setting(fam["setting"].iloc[0])
        ref = mv[(mv["model"] == model) & (mv["setting"] == ref_name)].set_index("problem_id")
        sevs = sorted(fam["severity"].unique())
        frames = [(s, fam[fam["severity"] == s].set_index("problem_id")) for s in sevs]
        pids = ref.index.intersection(frames[0][1].index)
        for _s, f in frames[1:]:
            pids = pids.intersection(f.index)
        if len(pids) == 0:
            continue
        ref_p = ref.loc[pids]
        fr_p = [(s, f.loc[pids]) for s, f in frames]
        for metric in METRICS:
            drops, vals = _curve(fr_p, ref_p, metric)
            boots = []
            for _ in range(n_boot):
                b = rng.choice(len(pids), len(pids))
                bp = pids[b]
                bd, bv = _curve([(s, f.loc[bp]) for s, f in fr_p], ref_p.loc[bp], metric)
                boots.append([_interp(bd, bv, t) for t in TARGETS])
            boots_a = np.asarray(boots, dtype=float)
            for k, t in enumerate(TARGETS):
                col = boots_a[:, k]
                col = col[np.isfinite(col)]
                rows.append({
                    "model": model, "family": family, "metric": metric, "target_drop": t,
                    "value": _interp(drops, vals, t),
                    "lo": float(np.quantile(col, 0.025)) if len(col) > 20 else np.nan,
                    "hi": float(np.quantile(col, 0.975)) if len(col) > 20 else np.nan,
                    "max_observed_drop": float(drops.max()), "n_settings": len(sevs), "n_problems": len(pids),
                })
    return pd.DataFrame(rows)
