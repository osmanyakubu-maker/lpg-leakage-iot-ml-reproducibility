"""Table 12: evaluation against design requirements R1-R12, assembled from the JSON outputs
of the other scripts and configs/design_requirements.yaml."""
import json
import operator

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

from lpgrisk.config import ROOT, cli, load_config, out_dir
from lpgrisk.plotting import save_table


def lookup(src, cfg, tdir):
    kind, _, rest = src.partition(":")
    if kind == "design":
        return None, "Designed, not measured"
    if kind == "measured":
        return rest, "Attained (implementation evidence outside this code)"
    if kind == "config":
        v = cfg
        for k in rest.split("."):
            v = v[k]
        return v, None
    if kind == "costconfig":
        env, key = rest.split(".")
        return yaml.safe_load(open(ROOT / "configs" / "cost_benefit.yaml"))["environments"][env][key], None
    if kind == "results":
        fname, key = rest.split(":", 1)
        p = tdir / fname
        if not p.exists():
            return None, f"Missing ({fname} not produced)"
        v = json.load(open(p))
        for k in key.split("."):
            v = v[k]
        return v, None
    raise ValueError(src)


def main():
    cfg = load_config(cli(__doc__).config)
    tdir = out_dir(cfg, "tables")
    reqs = yaml.safe_load(open(ROOT / "configs" / "design_requirements.yaml"))["requirements"]
    ops = {"ge": operator.ge, "lt": operator.lt}
    rows = []
    for rid, r in reqs.items():
        val, status = lookup(r["source"], cfg, tdir)
        if status is None:
            if r["op"] == "impl":
                status = "Attained" if val else "Not attained"
            elif val is None:
                status = "Missing"
            else:
                status = "Attained" if ops[r["op"]](float(val), r["threshold"]) else "Not met"
        rows.append({"Requirement": f"{rid}: {r['name']}", "Target": r["target"],
                     "Attained": "—" if val is None else (f"{val:.3f}" if isinstance(val, float) else str(val)),
                     "Status": status})
    df = pd.DataFrame(rows)
    save_table(df, tdir / "table12_requirements")
    print(df.to_string())


if __name__ == "__main__":
    main()
