"""Compare the regenerated results with the values reported in manuscript r10.0.6.

Reads the JSON/CSV outputs written by the other scripts and writes
results/reproduction_report.md and results/reproduction_report.json.

Status per item:
  Match         |reproduced - reported| <= tolerance
  Close         within 3 x tolerance
  Differs       further away
  Not produced  the output file or value is missing (e.g. the input data are not in place)

Tolerances are absolute, in the units of the reported value, and allow for seed-to-seed
variation and rounding in the manuscript. Hardware-dependent values (training time) are
listed for information only.
"""
import json
import math

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from lpgrisk.config import ROOT, cli, load_config


def _load(tdir, name):
    p = tdir / name
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.load(open(p))
    return pd.read_csv(p)


def _dig(obj, path):
    for k in path:
        if obj is None:
            return None
        if isinstance(obj, list) and isinstance(k, int):
            obj = obj[k] if k < len(obj) else None
        elif isinstance(obj, dict):
            obj = obj.get(k)
        else:
            return None
    return obj


# (group, item, reported, tolerance, getter(tables) -> value)
def checks():
    def j(name, *path, scale=1.0):
        def g(t):
            v = _dig(t(name), path)
            return None if v is None else float(v) * scale
        return g

    def mean_per_seed(name, model, key):
        def g(t):
            ps = _dig(t(name), [model, "per_seed"])
            vals = [m.get(key) for m in ps or [] if m.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        return g

    def table6(cfg_name, col="F1 (Critical)"):
        def g(t):
            df = t("table6_ablation.csv")
            if df is None:
                return None
            r = df[df["Configuration"] == cfg_name]
            return None if r.empty or pd.isna(r[col].iloc[0]) else abs(float(r[col].iloc[0]))
        return g

    def diag(fn):
        def g(t):
            cm = _dig(t("fig3_confusion_matrix.json"), ["normalised_confusion_mean"])
            return None if cm is None else float(fn(np.diag(np.array(cm))))
        return g

    def shap_share(feature):
        def g(t):
            recs = t("fig4_shap_importance.json")
            for r in recs or []:
                if r["Feature"] == feature:
                    return float(r["Share (%)"])
            return None
        return g

    def roi(env):
        def g(t):
            rows = _dig(t("table10_cost_benefit.json"), ["table"])
            for r in rows or []:
                if r["Environment"] == env:
                    return next(float(v) for k, v in r.items() if k.endswith("as Table 10 (%)"))
            return None
        return g

    def attained(t):
        df = t("table12_requirements.csv")
        return None if df is None else float((df["Status"] == "Attained").sum())

    def sens_max(param):
        def g(t):
            return _dig(t("table2_sensitivity.json"), ["summary", param, "max_abs_change_pp"])
        return g

    T3 = "table3_model_comparison.json"
    CH = "chamber_metrics.json"
    return [
        ("Table 3 / Sec. 4.1", "Ensemble accuracy", 0.981, 0.005, j(T3, "Proposed ensemble", "accuracy_mean")),
        ("Table 3 / Sec. 4.1", "Ensemble Critical F1", 0.970, 0.005, j(T3, "Proposed ensemble", "f1_critical_mean")),
        ("Table 3 / Sec. 4.1", "Ensemble AUC-ROC (macro)", 0.992, 0.005, mean_per_seed(T3, "Proposed ensemble", "auc_roc_macro_ovr")),
        ("Table 3 / Sec. 4.1", "Ensemble macro precision", 0.978, 0.005, mean_per_seed(T3, "Proposed ensemble", "precision_macro")),
        ("Table 3 / Sec. 4.1", "Ensemble macro recall", 0.971, 0.005, mean_per_seed(T3, "Proposed ensemble", "recall_macro")),
        ("Table 3 / Sec. 4.1", "Ensemble macro F1", 0.976, 0.005, mean_per_seed(T3, "Proposed ensemble", "f1_macro")),
        ("Abstract / Sec. 4.1", "Accuracy gain over XGBoost (pp)", 1.9, 0.3, j(T3, "proposed_vs_xgboost", "accuracy", "mean_diff", scale=100)),
        ("Abstract / Sec. 4.1", "Critical F1 gain over XGBoost", 0.024, 0.005, j(T3, "proposed_vs_xgboost", "f1_critical", "mean_diff")),
        ("Abstract / Sec. 4.1", "Cohen's d vs XGBoost (Critical F1, pooled)", 1.18, 0.15, j(T3, "proposed_vs_xgboost", "f1_critical", "d_pooled")),
        ("Sec. 4.1.1", "Balanced accuracy", 0.9712, 0.005, j("table7_summary.json", "balanced_accuracy")),
        ("Table 7", "Weighted precision", 0.980, 0.005, j("table7_summary.json", "precision_weighted")),
        ("Table 7", "Weighted recall", 0.981, 0.005, j("table7_summary.json", "recall_weighted")),
        ("Table 7", "Weighted F1", 0.980, 0.005, j("table7_summary.json", "f1_weighted")),
        ("Fig. 3", "Confusion-matrix diagonal, minimum", 0.958, 0.005, diag(np.min)),
        ("Fig. 3", "Confusion-matrix diagonal, maximum", 0.988, 0.005, diag(np.max)),
        ("Fig. 4", "SHAP share, LPG concentration (%)", 42, 3, shap_share("G")),
        ("Fig. 4", "SHAP share, LPG rate of change (%)", 18, 3, shap_share("dG_dt")),
        ("Fig. 4", "SHAP share, smoke (%)", 15, 3, shap_share("S")),
        ("Table 2", "Sensitivity: max |ΔF1| over λ_ewc sweep (pp)", 0.3, 0.3, sens_max("lambda_ewc")),
        ("Table 2", "Sensitivity: max |ΔF1| over SMOTE k sweep (pp)", 0.2, 0.2, sens_max("smote_k")),
        ("Table 6", "Without online learning, Critical F1", 0.948, 0.005, table6("Without online learning")),
        ("Table 6", "Without sensor fusion, Critical F1", 0.901, 0.005, table6("Without sensor fusion")),
        ("Table 6", "Without ensemble, Critical F1", 0.946, 0.005, table6("Without ensemble")),
        ("Table 6", "Without feature engineering, Critical F1", 0.928, 0.005, table6("Without feature engineering")),
        ("Table 6", "Without SMOTE/cost-sensitive, Critical F1", 0.936, 0.005, table6("Without SMOTE/cost-sensitive")),
        ("Table 6", "Cohen's d, without sensor fusion", 2.31, 0.25, table6("Without sensor fusion", "Cohen's d (pooled, vs full)")),
        ("Table 4", "Edge latency, mean (ms)", 65, 3, j("table4_latency.json", "edge_latency_ms_mean")),
        ("Table 4", "Edge latency, 95th pct (ms)", 120, 6, j("table4_latency.json", "edge_latency_ms_p95")),
        ("Table 4", "End-to-end latency, mean (ms)", 80.2, 2, j("table4_latency.json", "e2e_latency_ms_mean")),
        ("Table 4", "End-to-end latency, 95th pct (ms)", 270, 10, j("table4_latency.json", "e2e_latency_ms_p95")),
        ("Table 4", "CPU usage, 5th pct (%)", 25, 2, j("table4_latency.json", "cpu_pct_p5_p95", 0)),
        ("Table 4", "CPU usage, 95th pct (%)", 35, 2, j("table4_latency.json", "cpu_pct_p5_p95", 1)),
        ("Table 4", "RAM usage, 5th pct (MB)", 512, 25, j("table4_latency.json", "mem_mb_p5_p95", 0)),
        ("Table 4", "RAM usage, 95th pct (MB)", 768, 25, j("table4_latency.json", "mem_mb_p5_p95", 1)),
        ("Table 4", "Power (W)", 4.2, 0.1, j("table4_latency.json", "power_w_mean")),
        ("Table 4", "Bandwidth savings (%)", 89, 1, j("table4_latency.json", "bandwidth_savings", scale=100)),
        ("Table 5", "Sensing-to-actuation total (ms)", 420, 10, j("table5_timing_trace.json", "total_ms")),
        ("Sec. 4.4.1 / Table 8", "Chamber accuracy", 0.965, 0.005, j(CH, "accuracy")),
        ("Sec. 4.4.1 / Table 8", "Chamber Critical F1", 0.95, 0.01, j(CH, "f1_critical")),
        ("Sec. 4.4.1", "Chamber false-negative rate (%)", 1.4, 0.3, j(CH, "false_negative_rate", scale=100)),
        ("Sec. 4.4.1", "Chamber false-alarm rate (%)", 1.8, 0.3, j(CH, "false_alarm_rate", scale=100)),
        ("Sec. 4.4.1", "False-alarm rate without fusion (%)", 3.3, 0.3, j(CH, "false_alarm_rate_without_fusion", scale=100)),
        ("Sec. 4.4.1", "Fusion false-alarm reduction (%)", 45.5, 3, j(CH, "fusion_false_alarm_test", "relative_reduction", scale=100)),
        ("Sec. 4.4.1", "Fusion χ²", 12.4, 1.0, j(CH, "fusion_false_alarm_test", "chi2")),
        ("Sec. 4.4.1", "Packet delivery ratio (%)", 99.2, 0.1, j(CH, "packet_delivery_ratio", scale=100)),
        ("Sec. 4.2", "Lead time, mean (s)", 8.4, 0.5, j(CH, "lead_time_s", "mean")),
        ("Sec. 4.2", "Lead time, minimum (s)", 3.1, 0.5, j(CH, "lead_time_s", "min")),
        ("Sec. 4.2", "Lead time, maximum (s)", 14.6, 0.5, j(CH, "lead_time_s", "max")),
        ("Sec. 3.4.5", "Pruned (320 trees) Critical F1", 0.968, 0.005, j("edge_pruning_tinyml.json", "pruning", "f1_critical_pruned")),
        ("Sec. 3.4.5", "Pruning size reduction (%)", 36, 3, j("edge_pruning_tinyml.json", "pruning", "size_reduction", scale=100)),
        ("Sec. 4.4.3", "TinyML accuracy (%)", 95.5, 0.5, j("edge_pruning_tinyml.json", "tinyml", "accuracy_mean", scale=100)),
        ("Sec. 4.4.3", "TinyML model size (KB)", 184, 18, j("edge_pruning_tinyml.json", "tinyml", "estimated_model_kb")),
        ("Table 10", "ROI, Residential (%)", 173, 1, roi("Residential")),
        ("Table 10", "ROI, Restaurant (%)", 1036, 1, roi("Restaurant")),
        ("Table 10", "ROI, Hotel (%)", 581, 1, roi("Hotel")),
        ("Table 10", "ROI, Industrial (%)", 1654, 1, roi("Industrial")),
        ("Sec. 4.6", "Residential payback at 60 % prevention (years)", 2.4, 0.1,
         j("table10_cost_benefit.json", "sensitivity", "residential_payback_at_low_prevention_years")),
        ("Fig. 5 / Sec. 4.5", "SUS with SHAP", 78, 0.5, j("fig5_human_factors.json", "sus", "shap")),
        ("Fig. 5 / Sec. 4.5", "Decision time with SHAP (s)", 3.2, 0.05, j("fig5_human_factors.json", "decision_time_s", "shap")),
        ("Fig. 5 / Sec. 4.5", "Decision accuracy with SHAP (%)", 92, 0.5, j("fig5_human_factors.json", "decision_accuracy", "shap", scale=100)),
        ("Table 12", "Requirements attained (of 12)", 9, 0, attained),
    ]


def status(rep, tol, val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "Not produced"
    diff = abs(val - rep)
    if diff <= tol + 1e-9:
        return "Match"
    return "Close" if diff <= 3 * tol + 1e-9 else "Differs"


def main():
    cfg = load_config(cli(__doc__).config)
    rdir = ROOT / cfg["paths"]["results_dir"]
    tdir = rdir / "tables"
    cache = {}

    def t(name):
        if name not in cache:
            cache[name] = _load(tdir, name)
        return cache[name]

    rows = []
    for group, item, rep, tol, get in checks():
        try:
            val = get(t)
        except (KeyError, TypeError, ValueError, StopIteration):
            val = None
        rows.append({"source": group, "item": item, "reported": rep, "reproduced": val,
                     "tolerance": tol, "status": status(rep, tol, val)})

    train_s = _dig(t("table3_model_comparison.json"), ["Proposed ensemble", "train_time_s_mean"])
    counts = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    smoke = "smoke" in cfg["_config_path"]
    lines = ["# Reproduction report", "",
             f"Configuration: `{cfg['_config_path'].split('/')[-1]}`", ""]
    if smoke:
        lines += ["**Smoke test: these numbers come from the small test inputs and are not expected "
                  "to match the manuscript.**", ""]
    lines += ["Summary: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())), "",
              "| Source | Item | Reported | Reproduced | Tolerance | Status |", "|---|---|---|---|---|---|"]
    for r in rows:
        v = "—" if r["reproduced"] is None else f"{r['reproduced']:.4g}"
        lines.append(f"| {r['source']} | {r['item']} | {r['reported']:g} | {v} | ±{r['tolerance']:g} | {r['status']} |")
    lines += ["", "For information (hardware-dependent): mean training time "
              + ("not produced" if train_s is None else f"{train_s / 60:.1f} min")
              + " (manuscript: 42 min on a 16-core Xeon, 64 GB RAM).", ""]
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "reproduction_report.md").write_text("\n".join(lines))
    json.dump({"summary": counts, "items": rows, "train_time_s_mean": train_s},
              open(rdir / "reproduction_report.json", "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
