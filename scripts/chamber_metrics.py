"""Controlled test-chamber evaluation (Sec. 4.4.1, Sec. 4.2, Table 8):
accuracy, Critical F1, false-negative / false-alarm rates, false-alarm reduction from
Kalman fusion (chi-square), lead time, and packet delivery ratio."""
import copy

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from lpgrisk.config import cli, load_config, out_dir, resolve, save_json
from lpgrisk.metrics import alarm_rates, wilson_ci
from lpgrisk.pipeline import evaluate_proposed, prepare, train_proposed
from lpgrisk.plotting import save_table
from lpgrisk.stats import chi2_2x2


def lead_times(split, pred, cfg):
    """Per run: time the critical threshold is first crossed by the raw LPG reading minus
    the time the model first predicts >= High. Positive = warning before threshold."""
    trig, thr = cfg["evaluation"]["lead_time_trigger_class"], cfg["risk_rules"]["critical_ppm"]
    out = []
    for run in np.unique(split["groups"]):
        m = split["groups"] == run
        t, g, p = split["time"][m], split["lpg_raw"][m], pred[m]
        cross, warn = np.flatnonzero(g > thr), np.flatnonzero(p >= trig)
        if len(cross) and len(warn):
            out.append({"run_id": run, "lead_time_s": float(t[cross[0]] - t[warn[0]])})
    return out


def chamber_split_for_model(cfg, fusion, seed):
    ch = prepare(cfg, "chamber_samples", fusion=fusion)
    if cfg["evaluation"]["chamber_training"] == "simulation":
        sim = prepare(cfg, "simulation_samples", fusion=fusion)
        model = train_proposed(cfg, sim, seed)
        test = copy.copy(ch["test"]); test["X"] = sim["scaler"].transform(test["X_raw"])
    else:
        model = train_proposed(cfg, ch, seed)
        test = ch["test"]
    return model, test


def main():
    cfg = load_config(cli(__doc__).config)
    tdir = out_dir(cfg, "tables")
    amin = cfg["evaluation"]["alarm_min_class"]
    res = {"fused": [], "unfused": []}
    leads, first = [], {}
    for i, seed in enumerate(cfg["seeds"]["model_seeds"]):
        for fusion in (True, False):
            model, test = chamber_split_for_model(cfg, fusion, seed)
            met, aux = evaluate_proposed(cfg, model, test, adapt=True)
            ar = alarm_rates(test["y"], aux["pred"], amin)
            res["fused" if fusion else "unfused"].append({**{k: met[k] for k in ["accuracy", "f1_critical",
                                                          "fnr_within_critical"]}, **ar})
            if fusion:
                leads += [{**l, "seed": seed} for l in lead_times(test, aux["pred"], cfg)]
            if i == 0:
                first["fused" if fusion else "unfused"] = ar
        print(f"[seed {seed}] fused acc={res['fused'][-1]['accuracy']:.4f} "
              f"FPR fused={res['fused'][-1]['fpr']:.4f} unfused={res['unfused'][-1]['fpr']:.4f}")

    def mean(key, which="fused"):
        return float(np.mean([r[key] for r in res[which]]))

    u, f = first["unfused"], first["fused"]
    fa = chi2_2x2(u["fp"], u["fp"] + u["tn"], f["fp"], f["fp"] + f["tn"])
    fa["ci95_unfused"] = wilson_ci(u["fp"], u["fp"] + u["tn"])
    fa["ci95_fused"] = wilson_ci(f["fp"], f["fp"] + f["tn"])
    fa["note"] = ("First seed; sample-level test. Samples within a run are not independent, "
                  "so p-values are optimistic.")
    lt = pd.DataFrame(leads)
    timing = resolve(cfg, "chamber_timing")
    pdr = float(pd.read_csv(timing)["delivered"].mean()) if timing.exists() else None
    out = {
        "accuracy": mean("accuracy"), "f1_critical": mean("f1_critical"),
        "alarm_definition": f"positive = class >= {cfg['classes'][amin]}",
        "false_negative_rate": mean("fnr"), "false_alarm_rate": mean("fpr"),
        "false_alarm_rate_without_fusion": mean("fpr", "unfused"),
        "fnr_within_critical": mean("fnr_within_critical"),
        "fusion_false_alarm_test": fa,
        "lead_time_s": {"mean": float(lt["lead_time_s"].mean()) if len(lt) else None,
                        "min": float(lt["lead_time_s"].min()) if len(lt) else None,
                        "max": float(lt["lead_time_s"].max()) if len(lt) else None,
                        "n_run_seed_pairs": int(len(lt))},
        "packet_delivery_ratio": pdr,
        "per_seed": res,
    }
    save_json(out, tdir / "chamber_metrics.json")
    if len(lt):
        lt.to_csv(tdir / "lead_times.csv", index=False)

    # Table 8: simulation vs chamber
    import json
    sim_path = tdir / "table3_model_comparison.json"
    if sim_path.exists():
        sim = json.load(open(sim_path))["Proposed ensemble"]
        t8 = pd.DataFrame([
            {"Metric": "Overall accuracy", "Simulation": sim["accuracy_mean"], "Test chamber": out["accuracy"]},
            {"Metric": "Critical F1", "Simulation": sim["f1_critical_mean"], "Test chamber": out["f1_critical"]}])
        t8["Difference"] = t8["Test chamber"] - t8["Simulation"]
        save_table(t8, tdir / "table8_sim_vs_chamber")
        print(t8.to_string())
    print({k: v for k, v in out.items() if k != "per_seed"})


if __name__ == "__main__":
    main()
