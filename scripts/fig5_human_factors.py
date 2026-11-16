"""Fig. 5: effect of SHAP explanations on operator performance (Sec. 4.5).

Only de-identified aggregate data may be released (ethics approval GCU/RERC/2026/187), so this
script plots the aggregate means (with SD error bars where supplied) and reports the test
statistics computed by the authors on the individual-level data. It cannot recompute paired
tests from aggregates. Metrics whose means are missing are reported in the JSON but not plotted.
Expected file: data/human_factors/aggregate.csv (schema in data/human_factors/README.md)."""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from lpgrisk.config import cli, load_config, out_dir, resolve, save_json
from lpgrisk.plotting import plt

LABELS = {"sus": "SUS score", "decision_time_s": "Decision time (s)", "decision_accuracy": "Decision accuracy",
          "trust": "Trust", "confidence": "Confidence"}


def _f(v):
    return None if pd.isna(v) else float(v)


def main():
    cfg = load_config(cli(__doc__).config)
    df = pd.read_csv(resolve(cfg, "human_factors_aggregate"))
    out, plot = {}, []
    for m in LABELS:
        d = df[df["metric"] == m].set_index("condition")
        if not {"baseline", "shap"} <= set(d.index):
            continue
        s = d.loc["shap"]
        out[m] = {"baseline": _f(d.loc["baseline", "mean"]), "shap": _f(s["mean"]),
                  "sd_baseline": _f(d.loc["baseline", "sd"]), "sd_shap": _f(s["sd"]),
                  "p": _f(s["p_value"]), "p_reported": None if "p_reported" not in s or pd.isna(s["p_reported"])
                  else str(s["p_reported"]), "d": _f(s["cohens_d"]), "n": int(s["n"])}
        if out[m]["baseline"] is not None and out[m]["shap"] is not None:
            plot.append(m)
    fig, axes = plt.subplots(1, len(plot), figsize=(2.0 * len(plot), 2.8))
    for ax, m in zip(np.atleast_1d(axes), plot):
        o = out[m]
        sds = [o["sd_baseline"], o["sd_shap"]]
        yerr = None if any(v is None for v in sds) else sds
        ax.bar([0, 1], [o["baseline"], o["shap"]], yerr=yerr, color=["#9aa5b1", "#3b6ea5"], capsize=3)
        ax.set_xticks([0, 1], ["No SHAP", "SHAP"])
        ptxt = o["p_reported"] or (f"{o['p']:.3g}" if o["p"] is not None else "n/a")
        if ptxt[0] not in "<>=":
            ptxt = "=" + ptxt
        dtxt = f", d={o['d']:.2f}" if o["d"] is not None else ""
        ax.set_title(f"{LABELS[m]}\n(p{ptxt}{dtxt})", fontsize=8)
    fig.tight_layout()
    fdir = out_dir(cfg, "figures")
    fig.savefig(fdir / "fig5_human_factors.png"); fig.savefig(fdir / "fig5_human_factors.pdf")
    out["sus_with_shap"] = out.get("sus", {}).get("shap")
    out["not_plotted_missing_means"] = [m for m in out if m in LABELS and m not in plot]
    save_json(out, out_dir(cfg, "tables") / "fig5_human_factors.json")
    print(out)


if __name__ == "__main__":
    main()
