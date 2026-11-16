"""Data loading and leakage-safe grouped splitting (Sec. 3.8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .labels import derive_labels

REQUIRED = ["run_id", "time", "lpg", "temperature", "humidity", "smoke", "flame"]


def load_samples(path, cfg: dict) -> pd.DataFrame:
    """Load per-sample sensor data and rename columns to canonical names."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the dataset there (see data/README.md for the schema).")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    colmap = {v: k for k, v in cfg["columns"].items()}
    df = df.rename(columns=colmap)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df.sort_values(["run_id", "time"]).reset_index(drop=True)
    if "label" not in df.columns:
        df["label"] = derive_labels(df, cfg)
    df["label"] = df["label"].astype(int)
    return df


def grouped_split(df: pd.DataFrame, cfg: dict, seed: int) -> dict[str, np.ndarray]:
    """Split run IDs into train/val/test, stratified by the run's maximum class.

    Returns a dict of run-id arrays. No run appears in more than one split.
    """
    s = cfg["split"]
    run_cls = df.groupby("run_id")["label"].max()
    runs, strat = run_cls.index.values, run_cls.values
    # Stratification needs >= 2 runs per class in each partition; fall back otherwise.
    def _strat(y):
        vals, counts = np.unique(y, return_counts=True)
        return y if counts.min() >= 2 else None
    train, rest, _, y_rest = train_test_split(
        runs, strat, train_size=s["train"], random_state=seed, stratify=_strat(strat))
    val_frac = s["val"] / (s["val"] + s["test"])
    val, test = train_test_split(rest, train_size=val_frac, random_state=seed, stratify=_strat(y_rest))
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def subset(df: pd.DataFrame, runs) -> pd.DataFrame:
    return df[df["run_id"].isin(runs)].reset_index(drop=True)
