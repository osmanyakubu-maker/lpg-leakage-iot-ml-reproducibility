"""Configuration loading, seeding and output helpers."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike = "configs/default.yaml") -> dict:
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    cfg = yaml.safe_load(open(path))
    if "inherit" in cfg:
        parent = load_config(cfg.pop("inherit"))
        cfg = _deep_merge(parent, cfg)
    seeds = yaml.safe_load(open(ROOT / cfg["seeds_file"]))
    if "seeds_override" in cfg:
        seeds["model_seeds"] = cfg["seeds_override"]
    cfg["seeds"] = seeds
    cfg["baselines_grid"] = yaml.safe_load(open(ROOT / cfg["baselines_grid_file"]))
    cfg["_config_path"] = str(path)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # optional
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def resolve(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]


def out_dir(cfg: dict, sub: str) -> Path:
    p = ROOT / cfg["paths"]["results_dir"] / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2, default=default)


def cli(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/default.yaml")
    return ap.parse_args()
