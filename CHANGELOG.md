# Changelog

## 0.3.0 — for manuscript r10.0.6

- **Table 2:** `table2_sensitivity.py` regenerates the "Sensitivity observed" column (λ_ewc, δ,
  SMOTE k and cost-weight sweeps); sweep values in the `sensitivity` configuration block.
- **Reproduction report:** `check_results.py` compares every regenerated value with the value
  reported in the manuscript and writes `results/reproduction_report.{md,json}`; run last by
  `run_all.sh`.
- **Tests:** `test_manuscript_consistency.py` checks configuration and included data against the
  manuscript and that every file named in the documentation exists.
- **Environment:** `requirements-lock.txt` with exact tested versions; GitHub workflow runs the
  tests and the smoke test with `xgboost` and `shap` installed.
- Sec. 4.6 sensitivity reported value updated to approximately 29 months (2.4 years).
- Ethics approval number corrected to GCU/RERC/2026/187; Data Availability text aligned with the
  manuscript; repository URL added to `CITATION.cff`.

## 0.2.0 — for manuscript r10.0.6

- **Cost–benefit (Table 10):** residential annual savings updated from $20 to $120 (net of $30
  maintenance, Sec. 4.6). Each row now records whether its savings are already net of maintenance;
  the script prints the reported values next to the recomputed ones and the 60 % prevention
  sensitivity next to the manuscript's figure.
- **Human factors (Fig. 5):** `data/human_factors/aggregate.csv` now ships with the aggregate values
  reported in Sec. 4.5.
  `fig5_human_factors.py` handles missing SDs and means.
- **`run_all.sh`:** refuses real-data runs with the XGBoost stand-in or without `shap`; logs
  platform, CPU count and library availability; `PYTHON` selects the interpreter.
- **Table 3 script** records training wall time (Sec. 4.1 reports 42 min).
- **Table 4:** E2E latency definition documented (mean = filtering + features + inference + MQTT +
  actuator command, 80.2 ms); the smoke-test inputs follow it.
- **Deposit metadata:** `LICENSE` (MIT), `LICENSE-DOCS.md` (CC BY 4.0), `CITATION.cff`,
  `docs/results_map.md`, `docs/data_availability_statement.md`, `SHA256SUMS`.
- Tests added for the cost–benefit numbers.

## 0.1.0 — for manuscript r9.6

Initial release.
