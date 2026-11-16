#!/usr/bin/env bash
# Reproduce all computed tables and figures. Usage:
#   bash scripts/run_all.sh                              # real data, configs/default.yaml
#   bash scripts/run_all.sh configs/smoke_test.yaml      # smoke test
# Set PYTHON to choose the interpreter (default: python).
set -euo pipefail
CFG="${1:-configs/default.yaml}"
PY="${PYTHON:-python}"
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0

SMOKE=0; [[ "$CFG" == *smoke* ]] && SMOKE=1
HAVE_XGB=$("$PY" -c "import xgboost" 2>/dev/null && echo 1 || echo 0)
HAVE_SHAP=$("$PY" -c "import shap" 2>/dev/null && echo 1 || echo 0)
if [[ $SMOKE -eq 0 ]]; then
  # Guard: results for the manuscript must use the real libraries.
  if [[ "${LPG_ALLOW_XGB_FALLBACK:-0}" == "1" ]]; then
    echo "ERROR: LPG_ALLOW_XGB_FALLBACK=1 is for smoke tests only. Unset it for real runs." >&2; exit 2
  fi
  [[ $HAVE_XGB -eq 1 ]]  || { echo "ERROR: xgboost is required (pip install -r requirements.txt)." >&2; exit 2; }
  [[ $HAVE_SHAP -eq 1 ]] || { echo "ERROR: shap is required for Fig. 4 (pip install -r requirements.txt)." >&2; exit 2; }
fi

LOG_DIR=$("$PY" -c "import sys;sys.path.insert(0,'src');from lpgrisk.config import load_config;print(load_config('$CFG')['paths']['results_dir'])")
mkdir -p "$LOG_DIR"
{
  echo "config: $CFG"; date -u; git rev-parse HEAD 2>/dev/null || echo "git: n/a"
  "$PY" --version; "$PY" -c "import platform;print(platform.platform());import os;print('cpus:',os.cpu_count())"
  echo "xgboost installed: $HAVE_XGB; shap installed: $HAVE_SHAP; LPG_ALLOW_XGB_FALLBACK=${LPG_ALLOW_XGB_FALLBACK:-0}"
  "$PY" -m pip freeze 2>/dev/null | grep -Ei '^(numpy|pandas|scikit-learn|scipy|xgboost|shap|torch|matplotlib|pyyaml|emlearn)=' || true
} > "$LOG_DIR/run_environment.txt"

if [[ $SMOKE -eq 1 ]]; then "$PY" scripts/00_smoke_data_analysis.py; fi

run() { echo "=== $1"; "$PY" "scripts/$1" --config "$CFG" 2>&1 | tee "$LOG_DIR/log_${1%.py}.txt"; }
run table3_model_comparison.py      # Table 3 (+ predictions & model for Table 7, Fig 3, Fig 4)
run table7_fig3_per_class.py        # Table 7, Fig 3
if [[ $HAVE_SHAP -eq 1 ]]; then run fig4_shap.py; else echo "!!! shap not installed: Fig 4 skipped (smoke test only)"; fi
run table6_ablation.py              # Table 6
run table2_sensitivity.py           # Table 2, sensitivity column
run table4_5_latency.py             # Tables 4, 5
run chamber_metrics.py              # Sec 4.4.1, lead time (Sec 4.2), Table 8
run edge_pruning_tinyml.py          # Sec 3.4.5 pruning, Sec 4.4.3 TinyML
run table10_cost_benefit.py         # Table 10
run fig5_human_factors.py           # Fig 5
run table12_requirements.py         # Table 12
run check_results.py                # reproduced vs reported values -> reproduction_report.md
echo "Done. Outputs in $LOG_DIR/"
