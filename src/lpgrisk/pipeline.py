"""Shared pipeline: load -> denoise -> fuse -> features -> grouped split -> scale -> train/evaluate."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from .config import ROOT, resolve, set_seed
from .data import grouped_split, load_samples, subset
from .features import build_features
from .metrics import classification_metrics
from .models import StackedEnsemble
from .online import run_prequential
from .preprocessing import MinMaxScaler


def _cache_key(cfg, which, fusion, engineered):
    blob = json.dumps([cfg["paths"][which], cfg["preprocessing"], cfg["features"], cfg["kalman"],
                       cfg["risk_rules"], cfg["sampling_hz"], fusion, engineered], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def prepare(cfg: dict, which: str = "simulation_samples", fusion=True, engineered=True,
            split_seed: int | None = None) -> dict:
    df = load_samples(resolve(cfg, which), cfg)
    cache = ROOT / cfg["paths"]["results_dir"] / "cache" / f"{which}_{_cache_key(cfg, which, fusion, engineered)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        Xall, names = z["X"], list(z["names"])
    else:
        F = build_features(df, cfg, fusion=fusion, engineered=engineered)
        Xall, names = F.to_numpy(float), list(F.columns)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=Xall, names=np.array(names))
    seed = cfg["seeds"]["split_seed"] if split_seed is None else split_seed
    runs = grouped_split(df, cfg, seed)
    out = {"feature_names": names, "runs": runs, "df": df}
    masks = {k: df["run_id"].isin(v).to_numpy() for k, v in runs.items()}
    scaler = MinMaxScaler().fit(Xall[masks["train"]])          # fitted on training runs only
    for k, m in masks.items():
        out[k] = {"X": scaler.transform(Xall[m]), "X_raw": Xall[m], "y": df.loc[m, "label"].to_numpy(),
                  "groups": df.loc[m, "run_id"].to_numpy(), "time": df.loc[m, "time"].to_numpy(),
                  "lpg_raw": df.loc[m, "lpg"].to_numpy()}
    out["scaler"] = scaler
    return out


def train_proposed(cfg, data, seed, use_imbalance=True, bases=("rf", "xgb"), n_estimators=None):
    set_seed(seed)
    tr = data["train"]
    return StackedEnsemble(cfg, seed, use_imbalance=use_imbalance, bases=bases,
                           n_estimators=n_estimators).fit(tr["X"], tr["y"], tr["groups"])


def evaluate_proposed(cfg, model, split: dict, adapt=True) -> tuple[dict, dict]:
    """Evaluate on a split. With online learning enabled the split is streamed
    prequentially; the model is deep-copied so the caller's model is unchanged."""
    m = copy.deepcopy(model)
    if cfg["online"]["enabled"] and cfg["online"]["evaluation_mode"] == "prequential":
        r = run_prequential(m, split["X"], split["y"], cfg, adapt=adapt,
                            min_retrain=cfg["online"].get("min_retrain_samples", 2000))
        pred, proba = r["pred"], r["proba"]
        extra = {"n_meta_updates": r["n_updates"], "n_drift_retrains": len(r["drift_at"])}
    else:
        proba = m.predict_proba(split["X"]); pred = proba.argmax(1); extra = {}
    met = classification_metrics(split["y"], pred, proba, cfg["critical_class"])
    met.update(extra)
    return met, {"pred": pred, "proba": proba, "model": m}
