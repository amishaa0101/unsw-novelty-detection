"""Build the combo_<D> detectors from saved p-values of a finished run (no re-fitting).

Usage: python scripts/run_combo.py [--config configs/default.yaml]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.combo import run_combo  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = run_combo(ROOT / "outputs" / "runs" / Path(cfg["config_path"]).stem)
    print(f"Saved {len(paths)} combo run files")


if __name__ == "__main__":
    main()
