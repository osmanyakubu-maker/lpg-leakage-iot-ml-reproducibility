"""Two-level stacking ensemble: RF + XGBoost -> L2 logistic regression (Sec. 3.4.1, Eq. 10)."""
from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from .meta import SoftmaxRegression
from .smote import cost_weights, smote

N_CLASSES = 5


def make_xgb(params: dict, seed: int):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(objective="multi:softprob", num_class=N_CLASSES,
                             random_state=seed, eval_metric="mlogloss", **params)
    except ImportError:
        if os.environ.get("LPG_ALLOW_XGB_FALLBACK") == "1":
            warnings.warn("xgboost not installed: using sklearn HistGradientBoosting as a stand-in "
                          "(smoke testing only; results are NOT valid for the manuscript).")
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_iter=params.get("n_estimators", 100), max_depth=params.get("max_depth"),
                learning_rate=params.get("learning_rate", 0.1), random_state=seed)
        raise


def make_rf(params: dict, seed: int):
    return RandomForestClassifier(random_state=seed, **params)


def proba_full(model, X) -> np.ndarray:
    """predict_proba padded to all five classes (some folds may lack a class)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), N_CLASSES))
    out[:, np.asarray(model.classes_, int)] = p
    return out


def fit_base(model, X, y, cfg: dict, seed: int, use_imbalance: bool = True):
    imb = cfg["imbalance"]
    if use_imbalance and imb["smote"]:
        X, y, _ = smote(X, y, k=imb["smote_k"], sampling_strategy=imb["smote_sampling_strategy"], seed=seed)
    w = cost_weights(y, cfg["critical_class"], imb["critical_cost_weight"]) if use_imbalance else None
    model.fit(X, y, sample_weight=w)
    return model


class StackedEnsemble:
    """Base learners are trained on SMOTE-resampled, cost-weighted data. The meta-learner
    is trained on out-of-fold base-learner probabilities (grouped by run) of the
    original (not oversampled) samples, with the critical-class cost weight."""

    def __init__(self, cfg: dict, seed: int, use_imbalance=True, n_estimators=None, bases=("rf", "xgb")):
        self.cfg, self.seed, self.use_imbalance, self.bases = cfg, seed, use_imbalance, bases
        e = cfg["ensemble"]
        self.rf_params = dict(e["rf"]); self.xgb_params = dict(e["xgb"])
        if n_estimators:
            self.rf_params["n_estimators"] = self.xgb_params["n_estimators"] = n_estimators
        self.meta = SoftmaxRegression(N_CLASSES, l2=1.0 / e["meta"]["l2_C"])

    def _new_bases(self, seed):
        out = {}
        if "rf" in self.bases:
            out["rf"] = make_rf(self.rf_params, seed)
        if "xgb" in self.bases:
            out["xgb"] = make_xgb(self.xgb_params, seed)
        return out

    def _fit_bases(self, X, y, seed):
        return {k: fit_base(m, X, y, self.cfg, seed, self.use_imbalance) for k, m in self._new_bases(seed).items()}

    def base_proba(self, X, models=None) -> np.ndarray:
        models = models or self.base_models
        return np.hstack([proba_full(models[k], X) for k in self.bases])

    def fit(self, X, y, groups):
        folds = GroupKFold(n_splits=self.cfg["ensemble"]["stacking_cv_folds"])
        oof = np.zeros((len(X), N_CLASSES * len(self.bases)))
        for i, (tr, te) in enumerate(folds.split(X, y, groups)):
            m = self._fit_bases(X[tr], y[tr], self.seed + i + 1)
            oof[te] = self.base_proba(X[te], m)
        self.base_models = self._fit_bases(X, y, self.seed)
        w = cost_weights(y, self.cfg["critical_class"], self.cfg["imbalance"]["critical_cost_weight"]) \
            if self.use_imbalance else None
        self.meta.fit(oof, y, sample_weight=w)
        return self

    def refit_bases(self, X, y):
        """Retrain base learners from scratch (drift response, Algorithm 1 line 11)."""
        self.base_models = self._fit_bases(X, y, self.seed)

    def predict_proba(self, X):
        return self.meta.predict_proba(self.base_proba(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
