"""Baselines for Table 3, tuned by grid search on the validation split (Sec. 3.8)."""
from __future__ import annotations

import itertools
import warnings

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from .metrics import classification_metrics
from .models import fit_base, make_rf, make_xgb, proba_full


def grid(space: dict, max_configs: int, seed: int):
    space = {k: (v if isinstance(v, list) else [v]) for k, v in space.items()}
    keys = list(space)
    combos = [dict(zip(keys, v)) for v in itertools.product(*[space[k] for k in keys])]
    if len(combos) > max_configs:  # equal search budget across models
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in sorted(rng.choice(len(combos), max_configs, replace=False))]
    return combos


def _tabular_factory(name, params, seed):
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=seed, **params)
    if name == "SVM":
        p = {k: v for k, v in params.items() if k != "max_train_samples"}
        return SGDClassifier(random_state=seed, max_iter=50, tol=1e-4, **p)
    if name == "Random Forest":
        return make_rf({**params, "n_jobs": -1}, seed)
    if name == "XGBoost":
        return make_xgb({**params, "tree_method": "hist", "n_jobs": -1}, seed)
    raise KeyError(name)


TABULAR = {"Decision Tree": "decision_tree", "SVM": "svm", "Random Forest": "random_forest", "XGBoost": "xgboost"}


def _fit_eval(name, params, data, cfg, seed, split):
    tr = data["train"]
    X, y = tr["X"], tr["y"]
    cap = params.get("max_train_samples")
    if cap and len(y) > cap:
        idx = np.random.default_rng(seed).choice(len(y), cap, replace=False)
        X, y = X[idx], y[idx]
    m = fit_base(_tabular_factory(name, params, seed), X, y, cfg, seed, use_imbalance=True)
    ev = data[split]
    if hasattr(m, "predict_proba") and not (name == "SVM" and params.get("loss") == "hinge"):
        proba = proba_full(m, ev["X"])
    else:  # hinge SVM: no calibrated probabilities -> softmax of decision scores for AUC only
        s = m.decision_function(ev["X"])
        full = np.full((len(ev["X"]), 5), -1e9); full[:, m.classes_.astype(int)] = s
        e = np.exp(full - full.max(1, keepdims=True)); proba = e / e.sum(1, keepdims=True)
    return classification_metrics(ev["y"], proba.argmax(1), proba, cfg["critical_class"])


def tune_and_eval_tabular(name, data, cfg, seed) -> tuple[dict, dict]:
    g = cfg["baselines_grid"]
    best, best_score = None, -np.inf
    for params in grid(g[TABULAR[name]], g["max_configs"], seed):
        s = _fit_eval(name, params, data, cfg, seed, "val")[g["selection_metric"]]
        if s > best_score:
            best, best_score = params, s
    return _fit_eval(name, best, data, cfg, seed, "test"), best


# ---------------------------------------------------------------- sequence models
SEQ = {"LSTM": "lstm", "1D-CNN": "cnn1d", "Attentional Bi-LSTM": "attn_bilstm"}


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def tune_and_eval_sequence(name, data, cfg, seed):
    if not torch_available():
        warnings.warn(f"PyTorch not installed: skipping {name}.")
        return None, None
    from .baselines_dl import train_eval_sequence
    g = cfg["baselines_grid"]
    best, best_score = None, -np.inf
    for params in grid(g[SEQ[name]], g["max_configs"], seed):
        s = train_eval_sequence(name, params, data, cfg, seed, "val")[g["selection_metric"]]
        if s > best_score:
            best, best_score = params, s
    return train_eval_sequence(name, best, data, cfg, seed, "test"), best
