"""Tables 4 and 5: computational/communication latency and the sensing-to-actuation trace,
summarised from on-device timing logs (data/chamber/timing_trace.csv) and resource logs
(data/chamber/system_metrics.csv). These are measurements; this script only aggregates them."""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from lpgrisk.config import cli, load_config, out_dir, resolve, save_json
from lpgrisk.plotting import save_table

STAGES = [("sampling_ms", "Sampling"), ("filtering_ms", "Median + moving-average filtering"),
          ("feature_ms", "Feature extraction"), ("inference_ms", "Inference"),
          ("mqtt_rtt_ms", "MQTT round trip"), ("actuator_cmd_ms", "Actuator command"),
          ("valve_ms", "Valve mechanical actuation")]


def main():
    cfg = load_config(cli(__doc__).config)
    tdir = out_dir(cfg, "tables")
    tt = pd.read_csv(resolve(cfg, "chamber_timing"))
    sm = pd.read_csv(resolve(cfg, "chamber_system_metrics"))
    t4 = {
        "edge_latency_ms_mean": tt["edge_latency_ms"].mean(),
        "edge_latency_ms_p95": np.percentile(tt["edge_latency_ms"], 95),
        "e2e_latency_ms_mean": tt["e2e_latency_ms"].mean(),
        "e2e_latency_ms_p95": np.percentile(tt["e2e_latency_ms"], 95),
        "cpu_pct_p5_p95": [np.percentile(sm["cpu_pct"], 5), np.percentile(sm["cpu_pct"], 95)],
        "mem_mb_p5_p95": [np.percentile(sm["mem_mb"], 5), np.percentile(sm["mem_mb"], 95)],
        "power_w_mean": sm["power_w"].mean(),
        "bandwidth_savings": 1 - sm["bytes_sent"].sum() / sm["bytes_raw"].sum(),
        "n_events": len(tt),
    }
    save_json(t4, tdir / "table4_latency.json")
    save_table(pd.DataFrame([
        {"Metric": "Edge computation latency (mean)", "Value": f"{t4['edge_latency_ms_mean']:.0f} ms"},
        {"Metric": "Edge computation latency (95th pct)", "Value": f"{t4['edge_latency_ms_p95']:.0f} ms"},
        {"Metric": "End-to-end latency (mean)", "Value": f"{t4['e2e_latency_ms_mean']:.1f} ms"},
        {"Metric": "End-to-end latency (95th pct)", "Value": f"{t4['e2e_latency_ms_p95']:.0f} ms"},
        {"Metric": "CPU usage (5th-95th pct)", "Value": "{:.0f}-{:.0f}%".format(*t4["cpu_pct_p5_p95"])},
        {"Metric": "Memory usage (5th-95th pct)", "Value": "{:.0f}-{:.0f} MB".format(*t4["mem_mb_p5_p95"])},
        {"Metric": "Power consumption", "Value": f"{t4['power_w_mean']:.1f} W"},
        {"Metric": "Bandwidth savings", "Value": f"{100 * t4['bandwidth_savings']:.0f}%"},
    ]), tdir / "table4_latency")

    rows = [{"Stage": lab, "Mean (ms)": tt[c].mean(), "SD (ms)": tt[c].std()} for c, lab in STAGES]
    total = sum(r["Mean (ms)"] for r in rows)
    per_event_total = tt[[c for c, _ in STAGES]].sum(axis=1)
    rows.append({"Stage": "Total (sum of stage means)", "Mean (ms)": total, "SD (ms)": per_event_total.std()})
    save_table(pd.DataFrame(rows), tdir / "table5_timing_trace", float_fmt="{:.1f}")
    save_json({"total_ms": total, "per_event_total_p95_ms": float(np.percentile(per_event_total, 95))},
              tdir / "table5_timing_trace.json")
    print(t4); print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
