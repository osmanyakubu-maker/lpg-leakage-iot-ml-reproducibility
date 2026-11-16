"""Statistical tests used in the manuscript."""
from __future__ import annotations

import numpy as np
from scipy import stats


def cohens_d_pooled(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("inf")


def cohens_dz_paired(a, b) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("inf")


def paired_comparison(a, b) -> dict:
    """Seed-paired comparison of two models (same seeds, same test split)."""
    t = stats.ttest_rel(a, b)
    return {"mean_diff": float(np.mean(a) - np.mean(b)), "t": float(t.statistic), "p": float(t.pvalue),
            "d_pooled": cohens_d_pooled(a, b), "dz_paired": cohens_dz_paired(a, b), "n_seeds": len(a)}


def chi2_2x2(k1, n1, k2, n2) -> dict:
    """Chi-square test of two proportions (e.g. false alarms without vs with fusion)."""
    table = np.array([[k1, n1 - k1], [k2, n2 - k2]])
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    p1, p2 = k1 / n1, k2 / n2
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))  # Cohen's h for proportions
    return {"chi2": float(chi2), "p": float(p), "rate_1": p1, "rate_2": p2,
            "relative_reduction": (p1 - p2) / p1 if p1 else np.nan, "cohens_h": float(h)}
