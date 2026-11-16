"""On-line meta-learner adaptation with EWC and Page-Hinkley drift detection
(Sec. 3.4.2, Algorithm 1)."""
from __future__ import annotations

from collections import deque

import numpy as np

from .smote import cost_weights


class PageHinkley:
    """Page-Hinkley test on a stream of values (here: the 0/1 error indicator)."""

    def __init__(self, delta=0.005, threshold=50.0, alpha=0.9999):
        self.delta, self.threshold, self.alpha = delta, threshold, alpha
        self.reset()

    def reset(self):
        self.n, self.mean, self.m, self.M = 0, 0.0, 0.0, 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.m = self.alpha * self.m + (x - self.mean - self.delta)
        self.M = min(self.M, self.m)
        return (self.m - self.M) > self.threshold


def run_prequential(model, X: np.ndarray, y: np.ndarray, cfg: dict, adapt: bool = True,
                    min_retrain: int = 2000) -> dict:
    """Stream (X, y) in order: predict each sample, then reveal its label.

    adapt=False reproduces the "Without online learning" ablation (frozen meta-learner).
    Returns predictions, probabilities and a log of updates and drift events.
    """
    oc = cfg["online"]
    eta, lam, bs, every = oc["learning_rate"], oc["lambda_ewc"], oc["batch_size"], oc["update_every"]
    n_window = max(int(oc["window_days"] * 86400 * cfg["sampling_hz"]), bs)
    ph = PageHinkley(**oc["page_hinkley"])
    meta = model.meta
    Z = model.base_proba(X)                                      # meta-features
    W_prev = meta.W.copy()
    F = np.zeros_like(meta.W)                                     # Algorithm 1 line 1
    C = deque(maxlen=n_window)
    preds = np.empty(len(y), int)
    probs = np.empty((len(y), meta.K))
    n_updates, drifts, t = 0, [], 0
    crit, cw = cfg["critical_class"], cfg["imbalance"]["critical_cost_weight"]
    for i in range(len(y)):
        p = meta.predict_proba(Z[i:i + 1])[0]
        probs[i] = p
        preds[i] = p.argmax()
        if not adapt:
            continue
        C.append(i)                                               # line 3 (deque drops oldest, line 4)
        t += 1                                                    # line 5
        if t % every == 0:                                        # line 6
            idx = np.fromiter(C, int)[-bs:]                       # line 7: most recent 128
            g = meta.grad(Z[idx], y[idx], cost_weights(y[idx], crit, cw))
            # line 8: w <- w - eta*grad - lambda_ewc * F * (w - w_prev)
            meta.W = meta.W - eta * g - lam * F * (meta.W - W_prev)
            n_updates += 1
        if ph.update(float(preds[i] != y[i])) and len(C) >= min_retrain:   # line 10
            idx = np.fromiter(C, int)
            if len(np.unique(y[idx])) > 1:
                model.refit_bases(X[idx], y[idx])                 # line 11
                Z[i + 1:] = model.base_proba(X[i + 1:])
                Z[idx] = model.base_proba(X[idx])
                F = meta.fisher_diag(Z[idx], y[idx])              # line 12
                W_prev = meta.W.copy()
                drifts.append(i)
            ph.reset()                                            # line 13
    return {"pred": preds, "proba": probs, "n_updates": n_updates, "drift_at": drifts}
