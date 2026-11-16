# Data

The datasets are **not** included in this archive. Per the manuscript's Data Availability
statement, the simulation dataset, chamber logs, trained models and raw CFD outputs are
available from the corresponding author on reasonable request. The one exception is
`human_factors/aggregate.csv`, which is included: human-factors data are released only as
de-identified aggregates (ethics approval GCU/RERC/2026/187); see `human_factors/README.md`.

To reproduce the results, place the files below at these paths (or edit `configs/default.yaml`).

## `simulation/samples.csv` and `chamber/samples.csv` (one row per sensor sample, 2 Hz)

| column    | type   | meaning |
|-----------|--------|---------|
| `run_id`  | str    | CFD run or chamber run identifier (splits are grouped by this) |
| `time_s`  | float  | seconds since start of run |
| `lpg_ppm` | float  | LPG concentration (G) |
| `temp_c`  | float  | temperature (T) |
| `rh_pct`  | float  | relative humidity (H) |
| `smoke`   | float  | MQ-2 smoke reading (S) |
| `flame`   | 0/1    | IR flame detector (F) |
| `label`   | int 0-4 | ground-truth risk class from the three-level labelling procedure (Sec. 3.4.3). If absent, labels are derived from the rule thresholds in Sec. 3.2, which is only the fallback level of that procedure. |

`.parquet` is also accepted (requires `pyarrow`).

## `chamber/timing_trace.csv` (one row per alarm event; Tables 4-5, R5)

`event_id, sampling_ms, filtering_ms, feature_ms, inference_ms, mqtt_rtt_ms, actuator_cmd_ms, valve_ms, edge_latency_ms, e2e_latency_ms, delivered`

- Stage columns are the Table 5 stages, in milliseconds.
- `edge_latency_ms`: measured edge computation latency (Table 4, 65 ms mean / 120 ms p95).
- `e2e_latency_ms`: measured end-to-end latency excluding physical sensing and actuator
  execution (Table 4). In r10.0.6 its mean, 80.2 ms, equals filtering + feature extraction +
  inference + MQTT round trip + actuator command (15 + 12 + 6.2 + 38 + 9).
- `delivered`: 1 if the event's message was delivered (packet delivery ratio, R5 = 99.2 %).

## `chamber/system_metrics.csv` (periodic resource log; Table 4)

`cpu_pct, mem_mb, power_w, bytes_raw, bytes_sent`

## `tinyml/esp32_measurements.csv` (on-device measurements; Sec. 4.4.3)

`metric, value, unit` with metrics `peak_ram_kb, flash_mb, avg_power_w, inference_ms`.

## `human_factors/aggregate.csv` (included; de-identified aggregates; Fig. 5)

Schema: `human_factors/README.md`.

## `smoke/`

Created by `scripts/00_smoke_data_analysis.py` for the smoke test only. Do not report numbers from it.

## SB112

Not used by any script: the manuscript uses SB112 for descriptive context only (Sec. 4.4.2, Table 9).
Available at https://doi.org/10.5281/zenodo.6616632.
