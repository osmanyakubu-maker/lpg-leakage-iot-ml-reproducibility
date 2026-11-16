"""Table 10: cost-benefit analysis from configs/cost_benefit.yaml.

Payback = unit cost / annual savings.
Five-year ROI = (N * annual savings - unit cost) / unit cost, using the Table 10 savings. ROI net
of maintenance is also reported, deducting annual maintenance where the savings figure is gross
(see `maintenance_netted`).

Sensitivity (Sec. 4.6): annual savings scaled by 0.60 / 0.80 (60 % instead of 80 % prevention)."""
import _bootstrap  # noqa: F401
import pandas as pd
import yaml

from lpgrisk.config import ROOT, cli, load_config, out_dir, save_json
from lpgrisk.plotting import save_table


def payback_months(cost, s):
    return 12 * cost / s if s > 0 else float("inf")


def fmt_payback(m):
    return f"{m:.1f} months" if m < 24 else f"{m / 12:.1f} years"


def roi(cost, s, years):
    return 100 * (years * s - cost) / cost


def compute(cb: dict) -> dict:
    N = cb["horizon_years"]
    rows = []
    for env, v in cb["environments"].items():
        c, s, mnt = v["unit_cost"], v["annual_savings"], v["annual_maintenance"]
        s_net = s if v.get("maintenance_netted") else s - mnt
        rep = cb.get("reported", {}).get(env, {})
        rows.append({"Environment": env, "Cost per unit ($)": c, "Annual savings ($)": s,
                     "Payback": fmt_payback(payback_months(c, s)),
                     "Payback (reported)": rep.get("payback"),
                     f"ROI {N} y as Table 10 (%)": roi(c, s, N),
                     f"ROI {N} y (reported, %)": rep.get("roi_pct"),
                     f"ROI {N} y net of maintenance (%)": roi(c, s_net, N)})
    r = cb["environments"]["Residential"]
    f = cb["sensitivity"]["preventable_fraction_low"] / cb["assumptions"]["preventable_fraction"]
    sens = {"scale_factor": f,
            "residential_payback_at_low_prevention_years": r["unit_cost"] / (r["annual_savings"] * f),
            "reported": cb["sensitivity"].get("reported_residential_payback")}
    return {"table": rows, "sensitivity": sens}


def main():
    cfg = load_config(cli(__doc__).config)
    cb = yaml.safe_load(open(ROOT / "configs" / "cost_benefit.yaml"))
    res = compute(cb)
    df = pd.DataFrame(res["table"])
    save_table(df, out_dir(cfg, "tables") / "table10_cost_benefit", float_fmt="{:.1f}")
    save_json(res, out_dir(cfg, "tables") / "table10_cost_benefit.json")
    print(df.to_string()); print(res["sensitivity"])


if __name__ == "__main__":
    main()
