"""Load the YAML config, set the global seed, and record run information."""
import json
import platform
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Repo root = the folder that contains src/. All relative paths in configs are relative to it.
ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    """Read a YAML config. If it has a `base` key, start from that file and override it."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_name = cfg.pop("base", None)
    if base_name is not None:
        base = load_config(path.parent / base_name)
        base.update(cfg)  # top-level keys in the child replace the base keys
        cfg = base

    cfg["config_path"] = str(path)
    return cfg


def set_seed(seed):
    """One seed for every random step. Models also get it as random_state."""
    random.seed(seed)
    np.random.seed(seed)


def library_versions():
    import faiss
    import pandas
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "faiss": faiss.__version__,
    }


def run_info(cfg):
    """Seed, versions, time and config, saved next to every run's results."""
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "seed": cfg["seed"],
        "versions": library_versions(),
        "config": cfg,
    }


def save_run_info(cfg, name):
    """Write run info to outputs/logs/<name>_runinfo.json and return the path."""
    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"{name}_runinfo.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(run_info(cfg), f, indent=2)
    return out
