"""Table 3: model comparison on the simulation test split, mean ± SD over seeds.
Also: paired significance test of the proposed ensemble vs. XGBoost (Sec. 4.1 / Abstract).
Writes per-seed predictions of the proposed model (used by Table 7, Fig. 3) and the
first-seed trained model (used by Fig. 4 and the pruning check)."""
import pickle
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from lpgrisk.baselines import SEQ, TABULAR, tune_and_eval_sequence, tune_and_eval_tabular
from lpgrisk.config import cli, load_config, out_dir, save_json
from lpgrisk.pipeline import evaluate_proposed, prepare, train_proposed
from lpgrisk.plotting import save_table
from lpgrisk.stats import paired_comparison

COLS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_critical", "auc_roc_macro_ovr"]


def main():
    cfg = load_config(cli(__doc__).config)
    tdir, pdir, mdir = out_dir(cfg, "tables"), out_dir(cfg, "predictions"), out_dir(cfg, "models")
    data = prepare(cfg)
    per_seed = {}
    for i, seed in enumerate(cfg["seeds"]["model_seeds"]):
        print(f"[seed {seed}] proposed ensemble")
        t0 = time.perf_counter()
        model = train_proposed(cfg, data, seed)
        train_s = time.perf_counter() - t0
        met, aux = evaluate_proposed(cfg, model, data["test"], adapt=True)
        met["train_time_s"] = train_s            # hardware-dependent (Sec. 4.1 reports 42 min)
        per_seed.setdefault("Proposed ensemble", []).append(met)
        np.savez_compressed(pdir / f"proposed_seed{seed}.npz", y=data["test"]["y"], pred=aux["pred"],
                            proba=aux["proba"], time=data["test"]["time"], groups=data["test"]["groups"])
        if i == 0:
            pickle.dump({"model": model, "feature_names": data["feature_names"], "scaler": data["scaler"]},
                        open(mdir / "proposed_first_seed.pkl", "wb"))
        for name in TABULAR:
            print(f"[seed {seed}] {name}")
            m, best = tune_and_eval_tabular(name, data, cfg, seed)
            m["best_params"] = best
            per_seed.setdefault(name, []).append(m)
        for name in SEQ:
            m, best = tune_and_eval_sequence(name, data, cfg, seed)
            if m is not None:
                m["best_params"] = best
                per_seed.setdefault(name, []).append(m)

    rows, summary = [], {}
    order = list(TABULAR) + list(SEQ) + list(cfg["baselines_grid"]["contextual_baselines"]) + ["Proposed ensemble"]
    for name in order:
        if name in cfg["baselines_grid"]["contextual_baselines"]:
            c = cfg["baselines_grid"]["contextual_baselines"][name]
            rows.append({"Model": name, "accuracy_mean": c["accuracy"], "accuracy_sd": np.nan,
                         "f1_macro": c["f1_macro"], "source": c["source"] + " (cited, not re-run)"})
            continue
        if name not in per_seed:
            continue
        ms = per_seed[name]
        row = {"Model": name, "accuracy_mean": np.mean([m["accuracy"] for m in ms]),
               "accuracy_sd": np.std([m["accuracy"] for m in ms], ddof=1) if len(ms) > 1 else np.nan}
        for c in COLS[1:]:
            vals = [m.get(c, np.nan) for m in ms]
            row[c] = np.mean(vals)
            row[c + "_sd"] = np.std(vals, ddof=1) if len(ms) > 1 else np.nan
        row["source"] = f"this code, {len(ms)} seeds"
        rows.append(row)
        summary[name] = {"accuracy_mean": row["accuracy_mean"], "f1_critical_mean": row["f1_critical"],
                         "per_seed": ms}
        if name == "Proposed ensemble":
            summary[name]["train_time_s_mean"] = float(np.mean([m["train_time_s"] for m in ms]))
    table = pd.DataFrame(rows)
    save_table(table, tdir / "table3_model_comparison")

    if "XGBoost" in per_seed and len(per_seed["Proposed ensemble"]) > 1:
        a, b = per_seed["Proposed ensemble"], per_seed["XGBoost"]
        summary["proposed_vs_xgboost"] = {
            k: paired_comparison([m[k] for m in a], [m[k] for m in b]) for k in ["accuracy", "f1_critical"]}
    save_json(summary, tdir / "table3_model_comparison.json")
    print(table.to_string())


if __name__ == "__main__":
    main()
