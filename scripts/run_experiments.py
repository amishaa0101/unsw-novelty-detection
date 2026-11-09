"""Phase 7: the leave-one-attack-out loop.

Usage: python scripts/run_experiments.py --config configs/quick.yaml
"""
import argparse
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ROOT, load_config, save_run_info, set_seed  # noqa: E402
from src.experiment import run_round  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    name = Path(cfg["config_path"]).stem

    out_dir = ROOT / "outputs" / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(log_dir / f"experiments_{name}_{stamp}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)])
    logging.captureWarnings(True)
    warnings.filterwarnings("once")
    log = logging.getLogger("experiment")
    save_run_info(cfg, f"experiments_{name}")

    clean_path = ROOT / cfg["processed_dir"] / "clean.parquet"
    if not clean_path.exists():
        sys.exit(f"{clean_path} not found. Run: python scripts/run_audit.py")
    df = pd.read_parquet(clean_path)
    if cfg["subsample_frac"] < 1.0:
        # Stratified subsample for smoke tests: the same fraction of every class.
        df = df.groupby("attack_cat", group_keys=False).sample(frac=cfg["subsample_frac"], random_state=cfg["seed"])
        log.info("Subsampled to %d rows (frac %.2f)", len(df), cfg["subsample_frac"])
    df = df.reset_index(drop=True)
    log.info("Rows: %d. Classes: %s", len(df), df["attack_cat"].value_counts().to_dict())

    rows = []
    rounds = [(h, s) for h in cfg["attack_types"] for s in cfg["feature_settings"]]
    start = time.perf_counter()
    for hidden, setting in tqdm(rounds, desc="LOAO rounds"):
        log.info("Round: hidden=%s setting=%s", hidden, setting)
        t0 = time.perf_counter()
        rows += run_round(df, hidden, setting, cfg, out_dir)
        log.info("Round done in %.1fs", time.perf_counter() - t0)
    log.info("All rounds done in %.1fs", time.perf_counter() - start)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.4f")
    log.info("\n%s", summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
