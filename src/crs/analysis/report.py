"""Build every analysis table and figure from scored runs.

    python -m crs.run report --results results --out reports [--judged]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import figures
from .load import load_scored
from .matching import equal_loss_table
from .paired import fix_effects, vs_baseline
from .predict import failure_regression
from .trends import budget_curves, pruning_gain_vs_budget, severity_trends, voting_table


def _with_budget_fix(df: pd.DataFrame) -> pd.DataFrame:
    """Expose budget-truncated rows of fixes.yaml `budget` entries as ordinary fixes."""
    from .. import config

    extra = []
    for name, fx in config.fixes().items():
        if fx.type != "budget":
            continue
        b = fx.params["max_think"]
        rows = df[(df["budget"] == b) & (df["fix"] == f"none@{b}")].copy()
        rows["fix"] = name
        rows["budget"] = np.nan
        extra.append(rows)
    return pd.concat([df, *extra], ignore_index=True) if extra else df


def report(results: Path, out: Path, judged: bool = False) -> dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    df = load_scored(results, judged=judged)
    if df.empty:
        print(f"no scored runs under {results}; run `crs.run score` first")
        return {}
    df = _with_budget_fix(df)
    tables: dict[str, pd.DataFrame] = {}
    tables["vs_baseline"] = vs_baseline(df)
    tables["fix_effects"] = fix_effects(df)
    tables["equal_loss"] = equal_loss_table(df)
    tables["severity_trends"] = severity_trends(df)
    tables["failure_regression"] = failure_regression(df)
    curves = budget_curves(df)
    tables["budget_curves"] = curves
    tables["pruning_gain_vs_budget"] = pruning_gain_vs_budget(curves)
    tables["voting"] = voting_table(df)
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)

    if not tables["vs_baseline"].empty:
        figures.failure_vs_drop(tables["vs_baseline"], out)
        figures.severity_panels(tables["vs_baseline"], out)
    figures.outcome_bars(df, out)
    if not curves.empty:
        figures.budget_plot(curves, out)

    lines = [f"# Report ({'judge-confirmed' if judged else 'scan'} outcomes)", "",
             f"{len(df)} scored rows from {df.groupby(['model', 'setting', 'run']).ngroups} runs.", ""]
    vb = tables["vs_baseline"]
    if not vb.empty:
        cols = ["model", "setting", "family", "acc", "d_acc", "p_acc_mcnemar", "commit_fail", "commit_share_of_errors",
                "length_ratio", "hes_per_1k", "hes_per_1k_ref"]
        lines += ["## Compressed vs uncompressed (no fix)", "", vb[cols].round(3).to_markdown(index=False), ""]
    fe = tables["fix_effects"]
    if not fe.empty:
        cols = ["model", "setting", "fix", "d_acc", "p_acc_mcnemar", "d_commit_fail", "length_ratio"]
        lines += ["## Fix effects (vs no fix on the same version)", "", fe[cols].round(3).to_markdown(index=False), ""]
    (out / "summary.md").write_text("\n".join(lines))
    print(f"wrote {len(tables)} tables and figures to {out}")
    return tables
