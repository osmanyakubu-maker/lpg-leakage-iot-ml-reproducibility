"""Feature extraction (Eq. 3).

X = [G, T, H, S, F, dG/dt, dT/dt, dH/dt, mu_G,10s, sigma_G,10s, mu_S,10s, sigma_S,10s]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kalman import fuse
from .labels import rate_of_change
from .preprocessing import denoise

BASE = ["G", "T", "H", "S", "F"]
DERIV = ["dG_dt", "dT_dt", "dH_dt"]
WINDOW = ["mu_G_10s", "sd_G_10s", "mu_S_10s", "sd_S_10s"]


def feature_names(cfg: dict, fusion=True, engineered=True) -> list[str]:
    if not engineered:
        return BASE.copy()
    names = BASE + DERIV + WINDOW
    if cfg["features"].get("include_long_window"):
        names += ["mu_G_60s", "sd_G_60s", "mu_S_60s", "sd_S_60s"]
    return names


def build_features(df: pd.DataFrame, cfg: dict, fusion: bool = True,
                   engineered: bool = True) -> pd.DataFrame:
    """Return a feature frame aligned with df (same row order).

    fusion=False     -> denoised raw readings and finite-difference derivatives ("Without sensor fusion")
    engineered=False -> only the five instantaneous readings ("Without feature engineering")
    """
    d = denoise(df, cfg)
    if fusion and cfg["kalman"].get("enabled", True):
        st = fuse(d, cfg)
        X = pd.DataFrame(st, columns=BASE + DERIV)
    else:
        X = pd.DataFrame(d[["lpg", "temperature", "humidity", "smoke", "flame"]].to_numpy(), columns=BASE)
        for src, name in zip(["lpg", "temperature", "humidity"], DERIV):
            X[name] = rate_of_change(d, src)
    if not engineered:
        return X[BASE]
    X["run_id"] = df["run_id"].to_numpy()
    g = X.groupby("run_id", sort=False)
    hz = cfg["sampling_hz"]
    wins = [(cfg["features"]["short_window_s"], "10s")]
    if cfg["features"].get("include_long_window"):
        wins.append((cfg["features"]["long_window_s"], "60s"))
    for secs, tag in wins:
        w = int(secs * hz)
        for c in ["G", "S"]:
            X[f"mu_{c}_{tag}"] = g[c].transform(lambda s: s.rolling(w, min_periods=1).mean())
            X[f"sd_{c}_{tag}"] = g[c].transform(lambda s: s.rolling(w, min_periods=2).std()).fillna(0.0)
    return X[feature_names(cfg, fusion, engineered)]
