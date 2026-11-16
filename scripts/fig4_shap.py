"""Fig. 4: SHAP feature importance (Sec. 3.4.4, Sec. 4.1).

TreeSHAP is computed for each base learner of the first-seed model; mean |SHAP| per feature
(averaged over classes and samples) is normalised to percentages, then averaged across the
RF and XGBoost learners."""
import pickle

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from lpgrisk.config import cli, load_config, out_dir, save_json
from lpgrisk.pipeline import prepare
from lpgrisk.plotting import plt, save_table


def mean_abs_shap(explainer, X):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):                 # older shap: list of (n, f) per class
        sv = np.stack(sv, axis=-1)
    sv = np.abs(np.asarray(sv))
    if sv.ndim == 3:                         # (n, f, classes)
        return sv.mean(axis=(0, 2))
    return sv.mean(axis=0)


def main():
    import shap
    cfg = load_config(cli(__doc__).config)
    tdir, fdir, mdir = out_dir(cfg, "tables"), out_dir(cfg, "figures"), out_dir(cfg, "models")
    bundle = pickle.load(open(mdir / "proposed_first_seed.pkl", "rb"))
    model, names = bundle["model"], bundle["feature_names"]
    data = prepare(cfg)
    rng = np.random.default_rng(cfg["seeds"]["analysis_seed"])
    X = data["test"]["X"]
    X = X[rng.choice(len(X), min(2000, len(X)), replace=False)]
    shares = {}
    for k, m in model.base_models.items():
        imp = mean_abs_shap(shap.TreeExplainer(m), X)
        shares[k] = imp / imp.sum()
    share = np.mean(list(shares.values()), axis=0)
    df = pd.DataFrame({"Feature": names, "Share (%)": 100 * share,
                       **{f"{k} share (%)": 100 * v for k, v in shares.items()}}).sort_values("Share (%)", ascending=False)
    save_table(df, tdir / "fig4_shap_importance", float_fmt="{:.1f}")
    save_json(df.to_dict(orient="records"), tdir / "fig4_shap_importance.json")
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    d = df.iloc[::-1]
    ax.barh(d["Feature"], d["Share (%)"], color="#3b6ea5")
    ax.set_xlabel("Share of mean |SHAP| (%)")
    fig.tight_layout(); fig.savefig(fdir / "fig4_shap_importance.png"); fig.savefig(fdir / "fig4_shap_importance.pdf")
    print(df.to_string())


if __name__ == "__main__":
    main()
