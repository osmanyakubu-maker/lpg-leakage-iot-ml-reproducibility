"""Checks that the configuration and included data match manuscript r10.0.6, and that the
documentation, run script and code refer to the same files."""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lpgrisk.config import load_config  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/default.yaml")


def test_risk_rules_sec_3_2(cfg):
    r = cfg["risk_rules"]
    assert (r["low_ppm"], r["medium_ppm"], r["high_ppm"], r["critical_ppm"]) == (200, 500, 1500, 3000)
    assert r["critical_rate_ppm_per_s"] == 50 and r["flame_is_critical"] is True
    assert cfg["classes"] == ["Safe", "Low", "Medium", "High", "Critical"]


def test_preprocessing_and_features(cfg):
    assert cfg["preprocessing"]["median_window"] == 5          # five-point median filter (Sec. 3.2)
    assert cfg["features"]["short_window_s"] == 10             # Eq. 3
    assert cfg["kalman"]["enabled"] is True                    # Sec. 3.3


def test_split_sec_3_8(cfg):
    s = cfg["split"]
    assert (s["train"], s["val"], s["test"]) == (0.70, 0.15, 0.15)
    assert s["group_by"] == "run_id"


def test_seeds_sec_3_8(cfg):
    seeds = cfg["seeds"]["model_seeds"]
    assert len(seeds) == 10 and len(set(seeds)) == 10           # "means over 10 random seeds"


def test_table2_hyperparameters(cfg):
    assert cfg["online"]["lambda_ewc"] == 0.01
    assert cfg["online"]["page_hinkley"]["delta"] == 0.005
    assert cfg["imbalance"]["smote_k"] == 5
    assert cfg["imbalance"]["critical_cost_weight"] == 10
    assert cfg["ensemble"]["rf"]["n_estimators"] == cfg["ensemble"]["xgb"]["n_estimators"] == 500
    assert cfg["ensemble"]["pruned_n_estimators"] == 320
    assert cfg["online"]["learning_rate"] == 0.01
    assert cfg["online"]["batch_size"] == 128


def test_table2_sweeps_include_defaults(cfg):
    sc = cfg["sensitivity"]
    assert cfg["online"]["lambda_ewc"] in sc["lambda_ewc"]
    assert min(sc["lambda_ewc"]) == 0.001 and max(sc["lambda_ewc"]) == 0.1   # "0.001-0.1 tested"
    assert cfg["online"]["page_hinkley"]["delta"] in sc["ph_delta"]
    assert sc["smote_k"] == [3, 5, 7]
    assert sc["critical_cost_weight"] == [5, 10, 20]


def test_tinyml_int8(cfg):
    assert cfg["tinyml"]["feature_bits"] == 8                  # INT8 quantisation (Sec. 3.4.4)


def test_cost_benefit_inputs_sec_4_6():
    cb = yaml.safe_load(open(ROOT / "configs" / "cost_benefit.yaml"))
    env = cb["environments"]
    assert env["Residential"]["unit_cost"] == 220 and env["Residential"]["annual_maintenance"] == 30
    assert env["Industrial"]["unit_cost"] == 570 and env["Industrial"]["annual_maintenance"] == 80
    assert [env[e]["annual_savings"] for e in ["Residential", "Restaurant", "Hotel", "Industrial"]] == [120, 500, 300, 2000]
    assert cb["assumptions"]["preventable_fraction"] == 0.80
    assert cb["sensitivity"]["preventable_fraction_low"] == 0.60


def test_human_factors_aggregate_sec_4_5():
    df = pd.read_csv(ROOT / "data" / "human_factors" / "aggregate.csv").set_index(["metric", "condition"])
    assert (df.loc[("sus", "baseline"), "mean"], df.loc[("sus", "shap"), "mean"]) == (62, 78)
    assert (df.loc[("decision_time_s", "baseline"), "mean"], df.loc[("decision_time_s", "shap"), "mean"]) == (5.7, 3.2)
    assert (df.loc[("decision_accuracy", "baseline"), "mean"], df.loc[("decision_accuracy", "shap"), "mean"]) == (0.78, 0.92)
    d = {m: df.loc[(m, "shap"), "cohens_d"] for m in ["sus", "decision_time_s", "decision_accuracy", "trust", "confidence"]}
    assert d == {"sus": 1.24, "decision_time_s": 0.96, "decision_accuracy": 1.08, "trust": 0.82, "confidence": 0.91}
    assert (df["n"] == 20).all()


def test_twelve_design_requirements():
    reqs = yaml.safe_load(open(ROOT / "configs" / "design_requirements.yaml"))["requirements"]
    assert list(reqs) == [f"R{i}" for i in range(1, 13)]


def test_scripts_referenced_in_docs_exist():
    run_all = (ROOT / "scripts" / "run_all.sh").read_text()
    docs = run_all + (ROOT / "README.md").read_text() + (ROOT / "docs" / "results_map.md").read_text() \
        + (ROOT / "data" / "README.md").read_text()
    names = set(re.findall(r"([A-Za-z0-9_]+\.py)", docs))
    scripts = {p.name for d in ("scripts", "src/lpgrisk", "tests") for p in (ROOT / d).glob("*.py")}
    missing = sorted(n for n in names if n not in scripts and n != "_bootstrap.py")
    assert not missing, f"referenced but not in the archive: {missing}"
    for name in re.findall(r"^run (\S+\.py)", run_all, flags=re.M):
        assert (ROOT / "scripts" / name).exists(), name


def test_check_results_runs_without_outputs(tmp_path):
    """With no outputs present the checker reports every item as 'Not produced'."""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(f"inherit: configs/default.yaml\npaths:\n  results_dir: {tmp_path / 'res'}\n")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_results.py"), "--config", str(cfg_file)],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    report = (tmp_path / "res" / "reproduction_report.md").read_text()
    assert "Not produced" in report and "| Match |" not in report
