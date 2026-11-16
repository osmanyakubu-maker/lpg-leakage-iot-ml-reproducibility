# Reproducibility archive: LPG IoT/ML leakage-monitoring framework

Code, configuration files, random seeds and scripts for the computed tables and figures in

> Yakubu, O., Boateng, F. L., Kanzoni, I. D., Amponsah, A. A., Ahmed, M., Effah, E. A., & Adu-Boahen, N.
> *An IoT and Machine Learning Framework for LPG Leakage Monitoring and Automated Safety Control:
> A Design Science Research Study* (manuscript revision r10.0.6).

Archive version 0.3.0 (see `CHANGELOG.md`). Licence: MIT for code, CC BY 4.0 for documentation,
configuration and aggregate data (`LICENSE`, `LICENSE-DOCS.md`). Citation: `CITATION.cff`.
Repository: https://github.com/osmanyakubu-maker/lpg-leakage-iot-ml-reproducibility

## What reproduces what

`docs/results_map.md` maps every computed table and figure to its script, input files and output
files. In short:

| Script | Reproduces |
|---|---|
| `table3_model_comparison.py` | Table 3; ensemble vs XGBoost test (Abstract, Sec. 4.1); training time |
| `table7_fig3_per_class.py` | Table 7, Fig. 3, balanced accuracy and weighted precision (Sec. 4.1.1) |
| `fig4_shap.py` | Fig. 4 |
| `table6_ablation.py` | Table 6 |
| `table2_sensitivity.py` | Table 2, "Sensitivity observed" column (λ_ewc, δ, SMOTE k, cost weight) |
| `table4_5_latency.py` | Tables 4–5 (aggregates measured timing and resource logs) |
| `chamber_metrics.py` | Sec. 4.4.1 metrics, fusion false-alarm test, lead time (Sec. 4.2), Table 8 |
| `edge_pruning_tinyml.py` | 500 → 320 pruning (Sec. 3.4.5), ESP32 TinyML forest (Sec. 4.4.3) |
| `table10_cost_benefit.py` | Table 10 and the Sec. 4.6 sensitivity |
| `fig5_human_factors.py` | Fig. 5 (from the included de-identified aggregates) |
| `table12_requirements.py` | Table 12 |
| `check_results.py` | Compares every regenerated value with the manuscript → `results/reproduction_report.md` |

Figures 1, 2 and 6 are conceptual diagrams; Tables 1, 9 and 11 are descriptive. SB112 is used for
descriptive context only and is not read by any script.

Fig. 5 is drawn from the de-identified aggregates in `data/human_factors/aggregate.csv` (means and
test statistics; Sec. 4.5). Trust and confidence are recorded as test statistics (p, d) and appear
in the output JSON.

## Layout

```
configs/
  default.yaml                all hyperparameters, with manuscript section references
  seeds.yaml                  split seed + 10 model seeds + analysis seed
  baselines_grid.yaml         baseline search grids (equal budget)
  cost_benefit.yaml           Table 10 inputs and reported values
  design_requirements.yaml    R1–R12 targets and evidence sources (Table 12)
  smoke_test.yaml             fast installation check (+ baselines_grid_smoke.yaml)
requirements.txt              minimum versions; requirements-lock.txt: exact tested versions
src/lpgrisk/
  preprocessing.py            5-sample median + moving average, min–max (Eq. 2)
  kalman.py                   8-state Kalman fusion (Eqs. 4–9)
  features.py                 Eq. 3 feature vector
  labels.py                   rule-based risk classes (Sec. 3.2)
  data.py                     loading, run-grouped stratified 70/15/15 split (Sec. 3.8)
  smote.py                    SMOTE k=5 and critical-class cost weight 10 (Sec. 3.4.3)
  models.py, meta.py          RF(500)+XGBoost(500) → L2 logistic stacking (Sec. 3.4.1)
  online.py                   Algorithm 1: SGD + EWC meta-learner, Page–Hinkley drift retrain
  baselines.py, baselines_dl.py  Table 3 baselines with grid search on the validation split
  metrics.py, stats.py        metrics with explicit denominators; t-tests, Cohen's d, χ²
  pipeline.py, config.py, plotting.py   shared plumbing
scripts/                      one script per result (table above); run_all.sh runs them all;
                              00_smoke_data_analysis.py prepares the smoke-test inputs
data/                         README.md (schemas); human_factors/aggregate.csv (included)
docs/
  results_map.md              manuscript item → script → inputs → outputs
  data_availability_statement.md      Data Availability text
tests/
  test_smoke.py               unit tests for core components
  test_manuscript_consistency.py  configuration and included data vs the manuscript
.github/workflows/ci.yml      runs the tests and the smoke test on every push
```

## Running

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # or requirements-lock.txt + xgboost shap
# place the data as described in data/README.md
bash scripts/run_all.sh                         # → results/tables, results/figures
```

`run_all.sh` writes the Python version, platform and package versions to
`results/run_environment.txt`, and stops if `xgboost` or `shap` is missing. Its last step,
`check_results.py`, writes `results/reproduction_report.md`: a table of every reported value,
the regenerated value, the tolerance and a status (Match, Close, Differs or Not produced).

Smoke test, to check the installation (about 7 minutes on 2 cores). It runs every script on small
test inputs written to `data/smoke/`; the numbers it produces are not results:

```bash
python -m pytest tests/
LPG_ALLOW_XGB_FALLBACK=1 bash scripts/run_all.sh configs/smoke_test.yaml   # → results_smoke/
```

`LPG_ALLOW_XGB_FALLBACK=1` substitutes scikit-learn's HistGradientBoosting when `xgboost` is not
installed. It is accepted only with the smoke-test configuration. With `xgboost` and `shap`
installed the variable is not needed; this is how the GitHub workflow runs the smoke test.

Expected compute for the real run: the manuscript reports 42 minutes per full training on a
16-core Xeon with 64 GB RAM; Table 3 trains 10 seeds plus grid-searched baselines, and Table 6
retrains for each ablation, so allow several hours to days. The sequence baselines need PyTorch
(`torch`, optional in `requirements.txt`); without it they are skipped with a warning.

## Implementation details

Each of these is configurable:

- **Online learning in offline evaluation.** Test runs are streamed prequentially (predict, then
  update with the label). "Without online learning" (Table 6) freezes the meta-learner.
- **Algorithm 1, line 8:** w ← w − η∇L(w) − λ_ewc F (w − w_prev). The Fisher matrix F is computed
  at each drift-triggered retrain (line 12), so the EWC term acts from the first retrain onward.
- **Meta-learner loss.** Five-class (multinomial) logistic loss, Eq. 11; meta-features are out-of-fold
  base-learner probabilities from run-grouped folds.
- **Features.** The 12 features of Eq. 3 (10 s window statistics), all retained (Fig. 4). 60 s
  window statistics can be added with `features.include_long_window`.
- **"Without ensemble"** uses XGBoost as the only base learner, keeping the meta-learner and online updates.
- **Chamber false-negative / false-alarm rates** use a binary alarm view, positive = class ≥ Medium.
- **Chamber model** is trained on the 70 chamber training runs (`evaluation.chamber_training`).
- **Cohen's d** is reported both pooled (between models over seeds) and paired (d_z).
- **SVM** baseline uses a linear SGD SVM for tractability on 1.8 M samples.
- **iTransformer-XGBoost and TLPN** are reported as cited values from Zeng et al. [11], not re-run.
- **Statistical tests on chamber samples** treat samples as independent; samples within a run are
  correlated, so p-values are optimistic.
- **Table 10 ROI** is reported as in Table 10 and, in addition, net of maintenance for every row.
- **Table 2 sensitivity** varies one parameter at a time over the ranges in Table 2, on the first
  three seeds (`sensitivity` block of `configs/default.yaml`).

## Reproducibility notes

- Splits are grouped by `run_id`; the scaler is fitted on training runs only.
- All randomness is seeded from `configs/seeds.yaml`. Tree learners with `n_jobs=-1` are deterministic
  for a fixed seed; GPU training of the PyTorch baselines may not be bit-for-bit reproducible.
- Feature matrices are cached in `results/cache/`, keyed on the relevant configuration.
- `tests/test_manuscript_consistency.py` checks every configured value (risk thresholds, split,
  seeds, Table 2 hyperparameters, cost–benefit inputs, human-factors aggregates) against the
  manuscript, and that every file named in the documentation exists.
- `SHA256SUMS` lists checksums of every file in the archive (`sha256sum -c SHA256SUMS`).

## Contact

Corresponding author: Dr. Osman Yakubu, Garden City University, Kumasi, Ghana
(ORCID 0000-0001-5753-0537). Data requests: see the article's Data Availability statement.
