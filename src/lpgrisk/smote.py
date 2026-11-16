"""SMOTE (Chawla et al., 2002) with k nearest neighbours (Sec. 3.4.3, Eq. 12).

New sample: x_new = x_i + lambda * (x_nn - x_i), lambda ~ U(0, 1), so that x_new lies on
the segment between x_i and x_nn.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote(X: np.ndarray, y: np.ndarray, k: int = 5, sampling_strategy="auto",
          seed: int = 0, max_neighbour_pool: int = 200_000):
    """Oversample minority classes.

    sampling_strategy: "auto" -> every class up to the majority count;
                       float r -> every minority class up to r * majority count.
    Returns (X_res, y_res, is_oversampled).
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    n_max = counts.max()
    target = n_max if sampling_strategy == "auto" else int(float(sampling_strategy) * n_max)
    X_new, y_new = [X], [y]
    for c, n_c in zip(classes, counts):
        need = target - n_c
        if need <= 0 or n_c < 2:
            continue
        Xc = X[y == c]
        if len(Xc) > max_neighbour_pool:  # bound memory for very large classes
            Xc = Xc[rng.choice(len(Xc), max_neighbour_pool, replace=False)]
        kk = min(k, len(Xc) - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xc)
        base = rng.integers(0, len(Xc), need)
        _, idx = nn.kneighbors(Xc[base])
        pick = idx[np.arange(need), rng.integers(1, kk + 1, need)]
        lam = rng.random((need, 1))
        X_new.append(Xc[base] + lam * (Xc[pick] - Xc[base]))
        y_new.append(np.full(need, c))
    Xr, yr = np.vstack(X_new), np.concatenate(y_new)
    is_oversampled = np.zeros(len(yr), bool)
    is_oversampled[len(y):] = True
    return Xr, yr, is_oversampled


def cost_weights(y: np.ndarray, critical_class: int, weight: float) -> np.ndarray:
    w = np.ones(len(y))
    w[y == critical_class] = weight
    return w
