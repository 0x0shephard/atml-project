"""Figures for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAMILY_ORDER = ["wquant-g128", "wquant-pc", "wanda", "magnitude", "lowrank", "kvquant", "combo"]


def _families(t: pd.DataFrame) -> list[str]:
    fams = [f for f in FAMILY_ORDER if f in set(t["family"])]
    return fams + sorted(set(t["family"]) - set(fams))


def failure_vs_drop(vs_base: pd.DataFrame, out: Path) -> None:
    """Commit-fail rate against accuracy drop, one line per family: the equal-loss view."""
    for model, t in vs_base.groupby("model"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for fam in _families(t):
            f = t[t["family"] == fam].sort_values("severity")
            drop = -f["d_acc"].to_numpy()
            axes[0].plot(np.r_[0, drop], np.r_[f["commit_fail_ref"].iloc[0], f["commit_fail"]], "o-", label=fam)
            axes[1].plot(np.r_[0, drop], np.r_[np.nan, f["commit_share_of_errors"]], "o-", label=fam)
        axes[0].set(xlabel="accuracy drop vs uncompressed", ylabel="commitment failures / problems", title=model)
        axes[1].set(xlabel="accuracy drop vs uncompressed", ylabel="commitment failures / errors")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"failure_vs_drop_{model}.png", dpi=150)
        plt.close(fig)


def severity_panels(vs_base: pd.DataFrame, out: Path) -> None:
    metrics = [("acc", "accuracy"), ("length_ratio", "thinking length / uncompressed"),
               ("hes_per_1k", "hesitation words per 1k tokens"), ("commit_fail", "commitment failure rate")]
    for model, t in vs_base.groupby("model"):
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5))
        for fam in _families(t):
            f = t[t["family"] == fam].sort_values("severity")
            for ax, (m, lab) in zip(axes, metrics):
                ax.plot(f["severity"], f[m], "o-", label=fam)
                ax.set(xlabel="severity", ylabel=lab)
        axes[0].set_title(model)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"severity_{model}.png", dpi=150)
        plt.close(fig)


def outcome_bars(df: pd.DataFrame, out: Path) -> None:
    d = df[df["budget"].isna() & (df["fix"] == "none") & (df["samples"] == 1)]
    for model, g in d.groupby("model"):
        tab = g.groupby("setting")["outcome"].value_counts(normalize=True).unstack(fill_value=0)
        tab = tab.reindex(columns=[c for c in ["correct", "commit_fail", "never_correct"] if c in tab.columns])
        ax = tab.plot.bar(stacked=True, figsize=(max(6, 0.5 * len(tab)), 4), color=["#4c9f70", "#e07a5f", "#8d99ae"])
        ax.set(ylabel="share of problems", title=f"{model}: outcomes (no fix)")
        ax.figure.tight_layout()
        ax.figure.savefig(out / f"outcomes_{model}.png", dpi=150)
        plt.close(ax.figure)


def budget_plot(curves: pd.DataFrame, out: Path) -> None:
    for model, g in curves.groupby("model"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for setting, s in g.groupby("setting"):
            s = s.sort_values("budget")
            x = s["budget"].replace(np.inf, s["budget"][np.isfinite(s["budget"])].max() * 2 if np.isfinite(s["budget"]).any() else 1)
            ax.plot(x, s["acc"], "o-", label=setting)
        ax.set(xscale="log", xlabel="thinking budget (tokens; rightmost = full)", ylabel="accuracy", title=model)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(out / f"budget_{model}.png", dpi=150)
        plt.close(fig)
