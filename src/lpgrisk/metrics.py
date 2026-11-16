"""Classification metrics with explicit denominators (Sec. 4.1.1)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

N = 5


def classification_metrics(y, pred, proba=None, critical=4) -> dict:
    labels = list(range(N))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, labels=labels, average="weighted", zero_division=0)
    out = {
        "n_samples": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_macro": p.mean(), "recall_macro": r.mean(), "f1_macro": f.mean(),
        "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
        "f1_critical": f[critical], "recall_critical": r[critical], "precision_critical": p[critical],
        # FNR within the critical class: missed critical / all true critical.
        "fnr_within_critical": 1 - r[critical],
        "per_class": {"support": s.tolist(), "precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist()},
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
    }
    if proba is not None and len(np.unique(y)) == N:
        out["auc_roc_macro_ovr"] = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=labels)
    return out


def alarm_rates(y, pred, alarm_min_class: int) -> dict:
    """Binary alarm view: positive = class >= alarm_min_class.
    FNR = missed hazardous / all hazardous; FPR = false alarms / all non-hazardous."""
    t, a = np.asarray(y) >= alarm_min_class, np.asarray(pred) >= alarm_min_class
    tp, fn = int((t & a).sum()), int((t & ~a).sum())
    fp, tn = int((~t & a).sum()), int((~t & ~a).sum())
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "fnr": fn / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
            "fnr_over_all_samples": fn / max(len(t), 1)}


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / d
    return (c - h, c + h)
