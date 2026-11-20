"""Does hesitation predict the commitment failure? (RQ2)

Logistic regression of ``commit_fail`` on the standardised hesitation rate and the mean
entropy at hesitation points, with compression-family fixed effects and standard errors
clustered by problem (the same problem is solved by every version).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .load import main_view


def _z(x: pd.Series) -> pd.Series:
    s = x.std()
    return (x - x.mean()) / s if s and np.isfinite(s) else x * 0


def failure_regression(df: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    mv = main_view(df)
    mv = mv[(mv["fix"] == "none") & (mv["samples"] == 1)].copy()
    rows = []
    for model, d in mv.groupby("model"):
        d = d.dropna(subset=["hesitation_per_1k"]).copy()
        d["y"] = d["commit_fail"].astype(int)
        d["hes_z"] = _z(d["hesitation_per_1k"])
        d["ent_z"] = _z(d["mean_entropy_at_hesitation"].fillna(d["mean_entropy_at_hesitation"].mean()))
        d["len_z"] = _z(np.log1p(d["n_think_tokens"]))
        if d["y"].nunique() < 2:
            continue
        specs = {
            "hesitation": "y ~ hes_z + C(family)",
            "hesitation+entropy": "y ~ hes_z + ent_z + C(family)",
            "hesitation+entropy+length": "y ~ hes_z + ent_z + len_z + C(family)",
        }
        for name, f in specs.items():
            try:
                fit = smf.logit(f, d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d["problem_id"])[0]})
            except Exception as e:  # noqa: BLE001 - separation / singular design on small data
                rows.append({"model": model, "spec": name, "error": str(e)})
                continue
            for term in [t for t in fit.params.index if not t.startswith("C(") and t != "Intercept"]:
                rows.append({"model": model, "spec": name, "term": term, "coef": fit.params[term],
                             "odds_ratio": float(np.exp(fit.params[term])), "se": fit.bse[term], "p": fit.pvalues[term],
                             "n": int(fit.nobs), "pseudo_r2": fit.prsquared})
    return pd.DataFrame(rows)
