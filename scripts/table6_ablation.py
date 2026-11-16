"""Table 6: ablation study on the simulation test split (Critical-class F1, mean over seeds,
degradation vs. the full system, Cohen's d vs. full)."""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from lpgrisk.config import cli, load_config, out_dir, save_json
from lpgrisk.pipeline import evaluate_proposed, prepare, train_proposed
from lpgrisk.plotting import save_table
from lpgrisk.stats import cohens_d_pooled, paired_comparison

VARIANTS = {
    # name: (prepare kwargs, train kwargs, adapt)
    "Full system": ({}, {}, True),
    "Without online learning": ({}, {}, False),
    "Without sensor fusion": ({"fusion": False}, {}, True),
    "Without ensemble": ({}, {"bases": ("xgb",)}, True),          # XGBoost-only base level
    "Without feature engineering": ({"engineered": False}, {}, True),
    "Without SMOTE/cost-sensitive": ({}, {"use_imbalance": False}, True),
}


def main():
    cfg = load_config(cli(__doc__).config)
    tdir = out_dir(cfg, "tables")
    datasets, scores = {}, {k: [] for k in VARIANTS}
    for seed in cfg["seeds"]["model_seeds"]:
        full_model = None
        for name, (pk, tk, adapt) in VARIANTS.items():
            key = tuple(sorted(pk.items()))
            if key not in datasets:
                datasets[key] = prepare(cfg, **pk)
            data = datasets[key]
            if name == "Without online learning" and full_model is not None:
                model = full_model                                 # same trained model, frozen meta-learner
            else:
                model = train_proposed(cfg, data, seed, **tk)
            if name == "Full system":
                full_model = model
            met, _ = evaluate_proposed(cfg, model, data["test"], adapt=adapt)
            scores[name].append(met["f1_critical"])
            print(f"[seed {seed}] {name}: F1_critical={met['f1_critical']:.4f}")
    full = np.array(scores["Full system"])
    rows, js = [], {}
    for name, vals in scores.items():
        v = np.array(vals)
        r = {"Configuration": name, "F1 (Critical)": v.mean(),
             "SD": v.std(ddof=1) if len(v) > 1 else np.nan}
        if name != "Full system":
            r["Degradation (pp)"] = 100 * (v.mean() - full.mean())
            r["Degradation (relative %)"] = 100 * (v.mean() - full.mean()) / full.mean()
            r["Cohen's d (pooled, vs full)"] = cohens_d_pooled(full, v) if len(v) > 1 else np.nan
            if len(v) > 1:
                js[name] = paired_comparison(full, v)
        rows.append(r)
    table = pd.DataFrame(rows)
    save_table(table, tdir / "table6_ablation")
    save_json({"per_seed_f1_critical": scores, "tests_vs_full": js}, tdir / "table6_ablation.json")
    print(table.to_string())


if __name__ == "__main__":
    main()
