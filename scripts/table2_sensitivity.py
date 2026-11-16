"""Table 2, "Sensitivity observed" column: one-at-a-time sweeps of the hyperparameters in
Table 2 on the simulation test split.

- lambda_ewc and the Page-Hinkley delta act only on the online stage, so the trained model
  is reused and only the prequential evaluation is repeated.
- SMOTE k and the critical-class cost weight change base-learner training, so the model is
  retrained for each value.

Every other setting stays at its configs/default.yaml value. The sweep values and the
number of seeds are set in the `sensitivity` block of the configuration.
"""
import copy

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from lpgrisk.config import cli, load_config, out_dir, save_json
from lpgrisk.pipeline import evaluate_proposed, prepare, train_proposed
from lpgrisk.plotting import save_table

# parameter name -> (config path, retrain needed)
PARAMS = {
    "lambda_ewc": (("online", "lambda_ewc"), False),
    "ph_delta": (("online", "page_hinkley", "delta"), False),
    "smote_k": (("imbalance", "smote_k"), True),
    "critical_cost_weight": (("imbalance", "critical_cost_weight"), True),
}


def _get(cfg, path):
    for k in path:
        cfg = cfg[k]
    return cfg


def _set(cfg, path, value):
    c = copy.deepcopy(cfg)
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    return c


def main():
    cfg = load_config(cli(__doc__).config)
    sc = cfg["sensitivity"]
    seeds = cfg["seeds"]["model_seeds"][: sc["n_seeds"]]
    data = prepare(cfg)
    base_models = {s: train_proposed(cfg, data, s) for s in seeds}

    def score(c, model):
        met, _ = evaluate_proposed(c, model, data["test"], adapt=True)
        return met["f1_critical"], met["recall_critical"], met.get("n_drift_retrains", 0)

    reference = {s: score(cfg, base_models[s]) for s in seeds}
    ref_f1 = float(np.mean([reference[s][0] for s in seeds]))

    rows, raw = [], {}
    for name, (path, retrain) in PARAMS.items():
        default = _get(cfg, path)
        for value in sc[name]:
            c = _set(cfg, path, value)
            res = []
            for s in seeds:
                if value == default:
                    res.append(reference[s])
                    continue
                model = train_proposed(c, data, s) if retrain else base_models[s]
                res.append(score(c, model))
            f1 = np.array([r[0] for r in res])
            rows.append({"Parameter": name, "Value": value, "Default": value == default,
                         "F1 (Critical)": f1.mean(),
                         "Change vs default (pp)": 100 * (f1.mean() - ref_f1),
                         "Recall (Critical)": float(np.mean([r[1] for r in res])),
                         "Drift retrains (mean)": float(np.mean([r[2] for r in res]))})
            raw[f"{name}={value}"] = res
            print(f"{name}={value}: F1_critical={f1.mean():.4f} ({100 * (f1.mean() - ref_f1):+.2f} pp)")

    table = pd.DataFrame(rows)
    summary = {}
    for name in PARAMS:
        t = table[table["Parameter"] == name]
        summary[name] = {"max_abs_change_pp": float(t["Change vs default (pp)"].abs().max()),
                         "values": t["Value"].tolist()}
    tdir = out_dir(cfg, "tables")
    save_table(table, tdir / "table2_sensitivity")
    save_json({"seeds": seeds, "reference_f1_critical": ref_f1, "summary": summary,
               "per_seed": raw}, tdir / "table2_sensitivity.json")
    print(table.to_string())


if __name__ == "__main__":
    main()
