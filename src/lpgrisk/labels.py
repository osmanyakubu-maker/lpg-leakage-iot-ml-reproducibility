"""Rule-based risk classes (Sec. 3.2), used as the fallback labelling level (Sec. 3.4.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rate_of_change(df: pd.DataFrame, col: str) -> np.ndarray:
    """Per-run first difference divided by time step (units per second)."""
    d = df.groupby("run_id")[col].diff()
    dt = df.groupby("run_id")["time"].diff()
    return (d / dt).fillna(0.0).to_numpy()


def classify_rules(lpg, flame, dlpg_dt, rules: dict) -> np.ndarray:
    lpg = np.asarray(lpg, float)
    y = np.zeros(len(lpg), dtype=int)
    y[lpg >= rules["low_ppm"]] = 1
    y[lpg >= rules["medium_ppm"]] = 2
    y[lpg >= rules["high_ppm"]] = 3
    crit = lpg > rules["critical_ppm"]
    crit |= np.asarray(dlpg_dt) > rules["critical_rate_ppm_per_s"]
    if rules.get("flame_is_critical", True):
        crit |= np.asarray(flame) > 0.5
    y[crit] = 4
    return y


def derive_labels(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    return classify_rules(df["lpg"], df["flame"], rate_of_change(df, "lpg"), cfg["risk_rules"])
