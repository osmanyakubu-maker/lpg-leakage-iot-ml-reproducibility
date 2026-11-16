"""Unit tests for core components (no real data needed). Run: python -m pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpgrisk.config import load_config  # noqa: E402
from lpgrisk.data import grouped_split  # noqa: E402
from lpgrisk.kalman import build_model, kalman_run  # noqa: E402
from lpgrisk.labels import classify_rules  # noqa: E402
from lpgrisk.meta import SoftmaxRegression  # noqa: E402
from lpgrisk.metrics import classification_metrics  # noqa: E402
from lpgrisk.online import PageHinkley  # noqa: E402
from lpgrisk.smote import smote  # noqa: E402

CFG = load_config("configs/default.yaml")


def test_risk_rules():
    r = CFG["risk_rules"]
    y = classify_rules([100, 300, 800, 2000, 3500, 100, 100], [0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 60], r)
    assert y.tolist() == [0, 1, 2, 3, 4, 4, 4]


def test_split_is_grouped():
    df = pd.DataFrame({"run_id": np.repeat(np.arange(100), 10), "label": np.repeat(np.arange(100) % 5, 10)})
    s = grouped_split(df, CFG, seed=1)
    assert not set(s["train"]) & set(s["test"])
    assert not set(s["val"]) & set(s["test"])
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == 100


def test_smote_on_segment():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (200, 2)), rng.normal(5, 0.1, (10, 2))])
    y = np.r_[np.zeros(200, int), np.ones(10, int)]
    Xr, yr, is_oversampled = smote(X, y, k=5, seed=0)
    assert (yr == 1).sum() == 200
    new = Xr[is_oversampled]
    assert new.min() >= X[y == 1].min() - 1e-9 and new.max() <= X[y == 1].max() + 1e-9


def test_kalman_tracks_ramp():
    A, H, Q, R = build_model(0.5, CFG["kalman"]["process_noise_diag"], CFG["kalman"]["measurement_noise_diag"])
    t = np.arange(400) * 0.5
    Z = np.c_[100 + 10 * t, np.full(400, 25), np.full(400, 50), np.zeros(400), np.zeros(400)]
    Z[:, 0] += np.random.default_rng(0).normal(0, 5, 400)
    x = kalman_run(Z, A, H, Q, R)
    assert abs(x[-50:, 5].mean() - 10) < 1.0          # estimated dG/dt ~ 10 ppm/s


def test_meta_learner_and_fisher():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4)); y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 1)
    m = SoftmaxRegression(5, l2=1.0).fit(X, y)
    assert (m.predict(X) == y).mean() > 0.8
    assert m.fisher_diag(X, y).shape == m.W.shape


def test_page_hinkley_detects_shift():
    ph = PageHinkley(delta=0.005, threshold=20, alpha=1.0)
    fired = [ph.update(x) for x in np.r_[np.zeros(1000), np.ones(200)]]
    assert not any(fired[:1000]) and any(fired[1000:])


def test_metrics_denominators():
    y = np.array([4, 4, 4, 4, 0, 0, 0, 0, 0, 0]); p = np.array([4, 4, 4, 3, 0, 0, 0, 0, 0, 0])
    m = classification_metrics(y, p)
    assert abs(m["fnr_within_critical"] - 0.25) < 1e-12


def test_cost_benefit_reproduces_table10():
    import importlib.util
    import yaml
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("t10", root / "scripts" / "table10_cost_benefit.py")
    t10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t10)
    cb = yaml.safe_load(open(root / "configs" / "cost_benefit.yaml"))
    rows = {r["Environment"]: r for r in t10.compute(cb)["table"]}
    for env, rep in cb["reported"].items():
        assert abs(rows[env]["ROI 5 y as Table 10 (%)"] - rep["roi_pct"]) < 1.5, env
    assert rows["Residential"]["Payback"] == "22.0 months"

