"""Table 7 (per-class performance with sample counts) and Fig. 3 (normalised confusion matrix)
from the proposed model's test-split predictions written by table3_model_comparison.py.

Sample counts are those of the evaluated (test) split. Per-class metrics are means over seeds;
the confusion matrix is the mean of the row-normalised per-seed matrices."""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from lpgrisk.config import cli, load_config, out_dir, save_json
from lpgrisk.metrics import classification_metrics
from lpgrisk.plotting import plt, save_table


def main():
    cfg = load_config(cli(__doc__).config)
    pdir, tdir, fdir = out_dir(cfg, "predictions"), out_dir(cfg, "tables"), out_dir(cfg, "figures")
    mets, cms = [], []
    for seed in cfg["seeds"]["model_seeds"]:
        f = pdir / f"proposed_seed{seed}.npz"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: run table3_model_comparison.py first")
        z = np.load(f)
        m = classification_metrics(z["y"], z["pred"], z["proba"], cfg["critical_class"])
        mets.append(m)
        cm = np.array(m["confusion"], float)
        cms.append(cm / np.maximum(cm.sum(1, keepdims=True), 1))
    support = np.array(mets[0]["per_class"]["support"])
    P = np.mean([m["per_class"]["precision"] for m in mets], 0)
    R = np.mean([m["per_class"]["recall"] for m in mets], 0)
    F = np.mean([m["per_class"]["f1"] for m in mets], 0)
    rows = [{"Risk level": c, "Samples": int(n), "%": 100 * n / support.sum(), "Precision": p, "Recall": r, "F1": f}
            for c, n, p, r, f in zip(cfg["classes"], support, P, R, F)]
    w = support / support.sum()
    rows.append({"Risk level": "Macro average", "Samples": int(support.sum()), "%": 100.0,
                 "Precision": P.mean(), "Recall": R.mean(), "F1": F.mean()})
    rows.append({"Risk level": "Weighted average", "Samples": int(support.sum()), "%": 100.0,
                 "Precision": (w * P).sum(), "Recall": (w * R).sum(), "F1": (w * F).sum()})
    table = pd.DataFrame(rows)
    save_table(table, tdir / "table7_per_class")
    summary = {k: float(np.mean([m[k] for m in mets])) for k in
               ["accuracy", "balanced_accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
                "precision_macro", "recall_macro", "f1_macro", "fnr_within_critical"]}
    summary["n_test_samples"] = int(support.sum())
    save_json(summary, tdir / "table7_summary.json")
    print(table.to_string()); print(summary)

    cm = np.mean(cms, 0)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
    ax.set_xticks(range(5), cfg["classes"], rotation=30); ax.set_yticks(range(5), cfg["classes"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(fdir / "fig3_confusion_matrix.png"); fig.savefig(fdir / "fig3_confusion_matrix.pdf")
    save_json({"normalised_confusion_mean": cm}, tdir / "fig3_confusion_matrix.json")


if __name__ == "__main__":
    main()
