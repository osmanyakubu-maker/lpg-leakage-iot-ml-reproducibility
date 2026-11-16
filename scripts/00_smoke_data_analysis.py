"""Create the small input files used by configs/smoke_test.yaml.

These files exist only to check that the pipeline runs end to end. Nothing produced
from them may be reported as a result. Output goes to data/smoke/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "smoke"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(12345)
HZ = 2


def make_runs(n_runs, dur_s, prefix):
    rows = []
    for r in range(n_runs):
        n = dur_s * HZ
        t = np.arange(n) / HZ
        t0 = rng.uniform(0.1, 0.4) * dur_s
        cmax = rng.choice([150, 400, 1200, 2500, 5000, 9000], p=[.2, .2, .2, .15, .15, .1])
        tau = rng.uniform(20, 120)
        lpg = 50 + np.where(t > t0, cmax * (1 - np.exp(-(t - t0) / tau)), 0)
        lpg += rng.normal(0, 8, n)
        flame = np.zeros(n)
        if cmax >= 2500 and rng.random() < 0.3:
            flame[int(0.8 * n):] = 1
        smoke = 5 + 40 * flame + rng.normal(0, 2, n) + 0.002 * lpg
        temp = 25 + np.cumsum(rng.normal(0, 0.01, n)) + 3 * flame
        rh = 50 + np.cumsum(rng.normal(0, 0.02, n))
        rows.append(pd.DataFrame({"run_id": f"{prefix}{r:04d}", "time_s": t, "lpg_ppm": lpg.clip(0),
                                  "temp_c": temp, "rh_pct": rh, "smoke": smoke, "flame": flame}))
    return pd.concat(rows, ignore_index=True)


make_runs(80, 240, "cfd").to_csv(OUT / "simulation_samples.csv", index=False)
make_runs(40, 180, "ch").to_csv(OUT / "chamber_samples.csv", index=False)

n = 300
tt = pd.DataFrame({
    "event_id": range(n),
    "sampling_ms": rng.gamma(9, 2.5, n), "filtering_ms": rng.gamma(9, 1.8, n),
    "feature_ms": rng.gamma(9, 1.4, n), "inference_ms": rng.gamma(9, 0.8, n),
    "mqtt_rtt_ms": rng.gamma(6, 7, n), "actuator_cmd_ms": rng.gamma(6, 1.5, n),
    "valve_ms": rng.normal(300, 25, n), "delivered": (rng.random(n) > 0.01).astype(int)})
tt["edge_latency_ms"] = tt[["filtering_ms", "feature_ms", "inference_ms"]].sum(axis=1) + rng.gamma(4, 6, n)
# Table 4 end-to-end latency excludes physical sensing (sampling) and valve actuation:
# in r10.0.6 its mean (80.2 ms) equals filtering + features + inference + MQTT + actuator command.
tt["e2e_latency_ms"] = tt[["filtering_ms", "feature_ms", "inference_ms", "mqtt_rtt_ms",
                           "actuator_cmd_ms"]].sum(axis=1)
tt.to_csv(OUT / "timing_trace.csv", index=False)

pd.DataFrame({"cpu_pct": rng.uniform(20, 40, 500), "mem_mb": rng.uniform(450, 800, 500),
              "power_w": rng.normal(4, 0.2, 500), "bytes_raw": rng.integers(9000, 11000, 500),
              "bytes_sent": rng.integers(800, 1400, 500)}).to_csv(OUT / "system_metrics.csv", index=False)

pd.DataFrame({"metric": ["peak_ram_kb", "flash_mb", "avg_power_w", "inference_ms"],
              "value": [200.0, 1.0, 0.4, 6.0],
              "unit": ["KB", "MB", "W", "ms"]}).to_csv(OUT / "esp32_measurements.csv", index=False)

hf = []
for metric, base, shap in [("sus", 60, 75), ("decision_time_s", 6, 3.5), ("decision_accuracy", 0.75, 0.9),
                           ("trust", 3.2, 4.0), ("confidence", 3.1, 3.9)]:
    for cond, m in [("baseline", base), ("shap", shap)]:
        hf.append({"metric": metric, "condition": cond, "mean": m, "sd": abs(m) * 0.15, "n": 20,
                   "p_value": 0.01, "cohens_d": 1.0})
pd.DataFrame(hf).to_csv(OUT / "hf_aggregate.csv", index=False)
print("Smoke-test inputs written to", OUT, file=sys.stderr)
