"""Edge-layer noise reduction and normalisation (Sec. 3.2, Eq. 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSOR_COLS = ["lpg", "temperature", "humidity", "smoke", "flame"]


def denoise(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Causal 5-sample median filter followed by a causal moving average, per run.

    Causal (past-only) windows are used because the filter runs on-line at the edge.
    The flame channel is binary and is passed through unchanged.
    """
    p = cfg["preprocessing"]
    out = df.copy()
    g = out.groupby("run_id", sort=False)
    for c in ["lpg", "temperature", "humidity", "smoke"]:
        med = g[c].transform(lambda s: s.rolling(p["median_window"], min_periods=1).median())
        out[c] = med.groupby(out["run_id"], sort=False).transform(
            lambda s: s.rolling(p["moving_average_window"], min_periods=1).mean())
    return out


class MinMaxScaler:
    """x_norm = (x - x_min) / (x_max - x_min), fitted on training data only (Eq. 2)."""

    def fit(self, X: np.ndarray):
        self.min_ = X.min(axis=0)
        rng = X.max(axis=0) - self.min_
        self.range_ = np.where(rng == 0, 1.0, rng)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
