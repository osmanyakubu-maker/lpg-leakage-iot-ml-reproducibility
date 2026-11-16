"""Edge deployment checks:
  (a) Pruning 500 -> 320 trees (Sec. 3.4.5): Critical F1 and serialized model size, mean over seeds.
  (b) TinyML Random Forest for ESP32 (Sec. 4.4.3): accuracy of an INT8-input forest on the
      chamber test split, plus an estimated flash footprint. RAM, latency and power are
      on-device measurements and are read from data/tinyml/esp32_measurements.csv.
Optionally exports the forest to C with `emlearn` if installed."""
import pickle

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from lpgrisk.config import cli, load_config, out_dir, resolve, save_json, set_seed
from lpgrisk.metrics import classification_metrics
from lpgrisk.pipeline import evaluate_proposed, prepare, train_proposed
from lpgrisk.smote import cost_weights


def quantise(X, bits):
    q = (1 << bits) - 1
    return np.clip(np.round(np.clip(X, 0, 1) * q), 0, q).astype(np.uint8 if bits <= 8 else np.uint16)


def forest_bytes(rf):
    # per internal node: feature (1 B) + threshold (1 B) + left/right child (2 B each); per leaf: class (1 B)
    total = 0
    for est in rf.estimators_:
        t = est.tree_
        leaves = int((t.children_left == -1).sum())
        total += 6 * (t.node_count - leaves) + leaves
    return total


def main():
    cfg = load_config(cli(__doc__).config)
    tdir = out_dir(cfg, "tables")
    sim = prepare(cfg)
    e = cfg["ensemble"]
    rows = []
    for seed in cfg["seeds"]["model_seeds"]:
        for n in (e["rf"]["n_estimators"], e["pruned_n_estimators"]):
            m = train_proposed(cfg, sim, seed, n_estimators=n)
            met, _ = evaluate_proposed(cfg, m, sim["test"], adapt=True)
            rows.append({"seed": seed, "n_estimators": n, "f1_critical": met["f1_critical"],
                         "model_bytes": len(pickle.dumps(m.base_models))})
    pr = pd.DataFrame(rows).groupby("n_estimators")[["f1_critical", "model_bytes"]].mean()
    full_n, pruned_n = e["rf"]["n_estimators"], e["pruned_n_estimators"]
    pruning = {"f1_critical_full": pr.loc[full_n, "f1_critical"], "f1_critical_pruned": pr.loc[pruned_n, "f1_critical"],
               "delta_pp": 100 * (pr.loc[pruned_n, "f1_critical"] - pr.loc[full_n, "f1_critical"]),
               "size_reduction": 1 - pr.loc[pruned_n, "model_bytes"] / pr.loc[full_n, "model_bytes"]}
    print("pruning:", pruning)

    tc = cfg["tinyml"]
    ch = prepare(cfg, "chamber_samples")
    accs, sizes = [], []
    for seed in cfg["seeds"]["model_seeds"]:
        set_seed(seed)
        Xtr, Xte = quantise(ch["train"]["X"], tc["feature_bits"]), quantise(ch["test"]["X"], tc["feature_bits"])
        rf = RandomForestClassifier(n_estimators=tc["n_estimators"], max_depth=tc["max_depth"], random_state=seed, n_jobs=-1)
        rf.fit(Xtr, ch["train"]["y"], sample_weight=cost_weights(ch["train"]["y"], cfg["critical_class"],
                                                                   cfg["imbalance"]["critical_cost_weight"]))
        accs.append(classification_metrics(ch["test"]["y"], rf.predict(Xte))["accuracy"])
        sizes.append(forest_bytes(rf))
    tiny = {"accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
            "estimated_model_kb": float(np.mean(sizes) / 1024),
            "evaluated_on": "chamber test split"}
    try:
        import emlearn
        emlearn.convert(rf, method="inline").save(file=str(out_dir(cfg, "models") / "esp32_forest.h"), name="lpg_forest")
        tiny["c_export"] = "results/models/esp32_forest.h (last seed)"
    except ImportError:
        tiny["c_export"] = "emlearn not installed; C export skipped"
    meas = resolve(cfg, "tinyml_measurements")
    if meas.exists():
        tiny["on_device"] = pd.read_csv(meas).set_index("metric")["value"].to_dict()
    save_json({"pruning": pruning, "tinyml": tiny}, tdir / "edge_pruning_tinyml.json")
    print("tinyml:", tiny)


if __name__ == "__main__":
    main()
