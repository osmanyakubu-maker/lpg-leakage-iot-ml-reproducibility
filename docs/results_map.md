# Map from manuscript results to code, inputs and outputs (manuscript r10.0.6)

`results/` paths are created by `bash scripts/run_all.sh`. "On request" = available from the
corresponding author (Data Availability statement), placed at the path in `data/README.md`.

| Manuscript item | Reported values (r10.0.6) | Script | Input data | Output |
|---|---|---|---|---|
| Table 3 (model comparison, mean ± SD over 10 seeds) | Ensemble acc 0.981, Critical F1 0.970, AUC 0.992 | `scripts/table3_model_comparison.py` | `simulation/samples.csv` (on request) | `tables/table3_model_comparison.{csv,md,json}` |
| Abstract / Sec. 4.1: ensemble vs XGBoost (p, d) | +1.9 pp acc, +0.024 Critical F1, p < 0.001, d = 1.18 | `scripts/table3_model_comparison.py` | as above | `table3_model_comparison.json` → `proposed_vs_xgboost` |
| Sec. 4.1 training time | 42 min (16-core Xeon) | `scripts/table3_model_comparison.py` | as above | `table3_model_comparison.json` → `train_time_s_mean` (hardware-dependent) |
| Table 7, Sec. 4.1.1 (per-class, balanced acc., weighted precision) | balanced acc. 97.12 %, weighted precision 98.0 % | `scripts/table7_fig3_per_class.py` | predictions from Table 3 script | `tables/table7_per_class.*`, `table7_summary.json` |
| Fig. 3 (normalised confusion matrix) | diagonal 0.958–0.988 | `scripts/table7_fig3_per_class.py` | as above | `figures/fig3_confusion_matrix.{png,pdf}` |
| Fig. 4, Sec. 4.1 (SHAP importance) | LPG 42 %, rate 18 %, smoke 15 % | `scripts/fig4_shap.py` | first-seed model from Table 3 script | `figures/fig4_shap_importance.*`, `tables/fig4_shap_importance.*` |
| Table 6 (ablation) | Critical F1 0.948 / 0.901 / 0.946 / 0.928 / 0.936 | `scripts/table6_ablation.py` | `simulation/samples.csv` | `tables/table6_ablation.*` |
| Table 2, sensitivity column | λ_ewc 0.005–0.05: ±0.3 pp; k = 3–7: < 0.2 pp; cost weight ≥ 10 saturates Critical recall | `scripts/table2_sensitivity.py` | `simulation/samples.csv` | `tables/table2_sensitivity.*` |
| Table 4 (latency, CPU, RAM, power, bandwidth) | 65 / 120 ms edge; 80.2 / 270 ms E2E; 4.2 W; 89 % | `scripts/table4_5_latency.py` (aggregates logs) | `chamber/timing_trace.csv`, `chamber/system_metrics.csv` (on request) | `tables/table4_latency.*` |
| Table 5 (sensing-to-actuation trace) | total 420 ms | `scripts/table4_5_latency.py` | `chamber/timing_trace.csv` | `tables/table5_timing_trace.*` |
| Sec. 4.4.1 chamber metrics | acc 96.5 %, Critical F1 0.95, FNR 1.4 %, FPR 1.8 %, PDR 99.2 % | `scripts/chamber_metrics.py` | `chamber/samples.csv`, `chamber/timing_trace.csv` | `tables/chamber_metrics.json` |
| Sec. 4.4.1 fusion false-alarm reduction | 3.3 % → 1.8 %, 45.5 %, χ² = 12.4 | `scripts/chamber_metrics.py` | as above | `chamber_metrics.json` → `fusion_false_alarm_test` |
| Sec. 4.2 lead time | mean 8.4 s (3.1–14.6 s) | `scripts/chamber_metrics.py` | as above | `chamber_metrics.json` → `lead_time_s`, `tables/lead_times.csv` |
| Table 8 (simulation vs chamber) | 98.1 vs 96.5 %; 0.970 vs 0.95 | `scripts/chamber_metrics.py` | Table 3 output + chamber | `tables/table8_sim_vs_chamber.*` |
| Sec. 3.4.5 pruning 500 → 320 | Critical F1 0.970 → 0.968, −36 % size | `scripts/edge_pruning_tinyml.py` | `simulation/samples.csv` | `tables/edge_pruning_tinyml.json` → `pruning` |
| Sec. 4.4.3 TinyML on ESP32 | 95.5 % acc, 184 KB model; 212 KB RAM, 1.1 MB flash, 0.42 W, 6.2 ms measured on device | `scripts/edge_pruning_tinyml.py` | `chamber/samples.csv`, `tinyml/esp32_measurements.csv` (on request) | `edge_pruning_tinyml.json` → `tinyml` |
| Table 10, Sec. 4.6 (cost–benefit) | payback 22 / 5 / 9 / 3.5 months; ROI 173 / 1,036 / 581 / 1,654 % | `scripts/table10_cost_benefit.py` | `configs/cost_benefit.yaml` (included) | `tables/table10_cost_benefit.*` |
| Fig. 5, Sec. 4.5 (human factors) | SUS 62 → 78; 5.7 → 3.2 s; 78 → 92 % | `scripts/fig5_human_factors.py` | `human_factors/aggregate.csv` (**included**) | `figures/fig5_human_factors.*` |
| Table 12 (design requirements) | 9 of 12 attained | `scripts/table12_requirements.py` | outputs above + `configs/design_requirements.yaml` | `tables/table12_requirements.*` |
| All of the above | reported vs regenerated, with tolerances | `scripts/check_results.py` | outputs above | `reproduction_report.{md,json}` |

## Items with no generating code

- **Figs. 1, 2, 6**: conceptual diagrams.
- **Tables 1, 9, 11**: descriptive; no computed results.
- **SB112** (Sec. 4.4.2, Table 9): used for descriptive context only; no script reads it.
- **Sec. 3.5 / 4.4.1 power-failure test** and **R8, R10** (security, explainability): implementation or bench
  evidence outside this code base.
