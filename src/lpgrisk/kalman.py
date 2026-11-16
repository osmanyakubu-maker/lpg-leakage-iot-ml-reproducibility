"""Multi-sensor Kalman fusion (Sec. 3.3, Eqs. 4-9).

State   x_k = [G, T, H, S, F, dG/dt, dT/dt, dH/dt]^T
Measure z_k = [G, T, H, S, F]^T
Constant-velocity model for G, T and H; random walk for S and F. No control input (B u = 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_model(dt: float, q_diag, r_diag):
    A = np.eye(8)
    A[0, 5] = A[1, 6] = A[2, 7] = dt
    H = np.zeros((5, 8))
    H[np.arange(5), np.arange(5)] = 1.0
    return A, H, np.diag(q_diag), np.diag(r_diag)


def kalman_run(Z: np.ndarray, A, H, Q, R) -> np.ndarray:
    n = Z.shape[0]
    x = np.zeros(8)
    x[:5] = Z[0]
    P = np.eye(8) * 10.0
    I = np.eye(8)
    out = np.empty((n, 8))
    for k in range(n):
        x = A @ x                                  # Eq. 5
        P = A @ P @ A.T + Q                        # Eq. 6
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)             # Eq. 7
        x = x + K @ (Z[k] - H @ x)                 # Eq. 8
        P = (I - K @ H) @ P                        # Eq. 9
        out[k] = x
    return out


def fuse(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Run the filter independently on every run; returns an (n_samples, 8) array."""
    kc = cfg["kalman"]
    A, H, Q, R = build_model(1.0 / cfg["sampling_hz"], kc["process_noise_diag"], kc["measurement_noise_diag"])
    cols = ["lpg", "temperature", "humidity", "smoke", "flame"]
    out = np.empty((len(df), 8))
    for _, idx in df.groupby("run_id", sort=False).indices.items():
        out[idx] = kalman_run(df.loc[idx, cols].to_numpy(float), A, H, Q, R)
    return out
