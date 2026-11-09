"""Diagnostics for a small excess of the pooled false unknown rate above alpha. Facts only, changes nothing.

For each round of a finished run it recomputes the split, the preprocessing and the detectors
(deterministic: same seed, and for GMM the reg_covar the run chose), checks that the recomputed test
statistics T equal the saved ones, and reports:
  (a) known test rows whose feature vector is identical to a calib_final row's vector
      (raw feature columns, and after preprocessing, where dropped columns can make rows identical)
  (b) ties: distinct T values among calib_final, and known test rows whose T equals a calib_final T exactly
  (c) the pooled false unknown rate at each alpha, from the saved run file

Usage: python scripts/check_calibration_excess.py --config configs/default.yaml [--hidden DoS Worms]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import calibration as cal  # noqa: E402
from src.baseline import ConfidenceBaseline  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.data import feature_columns  # noqa: E402
from src.detectors import GMMDetector, KNNDetector, PCADetector, PerClassDetector  # noqa: E402
from src.experiment import class_names  # noqa: E402
from src.preprocess import Preprocessor  # noqa: E402
from src.split import make_splits  # noqa: E402


def row_keys(A):
    return {r.tobytes() for r in np.ascontiguousarray(A)}


def statistics(det, run, cfg, known, X, y, N):
    """Recompute T for calib_final and test exactly as the experiment did."""
    seed, ccfg = cfg["seed"], cfg["calibration"]
    if det == "baseline":
        clf = ConfidenceBaseline(seed).fit(X["train"], y["train"])
        return clf.predict(X["calib_final"], N)[0], clf.predict(X["test"], N)[0]
    if det == "gmm":
        g = cfg["gmm"]
        reg = run["reg_covar_chosen"]
        make = lambda: GMMDetector(g["k_grid"], g["covariance_type"], g["n_init"], reg, g["max_fit_rows"], seed)
    elif det == "pca":
        make = lambda: PCADetector(cfg["pca"]["explained_variance"])
    else:
        make = lambda: KNNDetector(cfg["knn"]["k"])
    pcd = PerClassDetector(make, known).fit(X["train"], y["train"])
    S = {k: pcd.score(X[k]) for k in ["calib_norm", "calib_final", "test"]}
    if det in ("pca", "knn"):
        S = {k: np.log(v + ccfg["log_eps"]) for k, v in S.items()}
    norm = cal.fit_normaliser(S["calib_norm"], y["calib_norm"], known, ccfg["min_scale"])
    return cal.combine(cal.normalise(S["calib_final"], norm), known)[0], cal.combine(cal.normalise(S["test"], norm), known)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--hidden", nargs="*", help="hidden classes to check (default: all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = ROOT / "outputs" / "runs" / Path(cfg["config_path"]).stem
    df = pd.read_parquet(ROOT / cfg["processed_dir"] / "clean.parquet").reset_index(drop=True)
    names = class_names(df)
    code = {n: i for i, n in enumerate(names)}
    y_all = df["attack_cat"].map(code).to_numpy()
    feats = feature_columns(df)
    rows = []
    for hidden in args.hidden or cfg["attack_types"]:
        H, N = code[hidden], code["Normal"]
        known = [code[n] for n in names if n != hidden]
        sp = make_splits(df, hidden, cfg["split"], cfg["seed"])
        test_known = sp["test"][y_all[sp["test"]] != H]
        raw_cf = row_keys(df.loc[sp["calib_final"], feats].astype(str).to_numpy())
        raw_same = sum(k in raw_cf for k in row_keys(df.loc[test_known, feats].astype(str).to_numpy()))
        for setting in cfg["feature_settings"]:
            prep = Preprocessor(setting, cfg["shortcut_features"], cfg["noise_features"], cfg["strict_extra_features"],
                                cfg["preprocess"]["proto_top_n"], cfg["preprocess"]["log1p"]).fit(df.loc[sp["train"]])
            X = {k: prep.transform(df.loc[idx]) for k, idx in sp.items()}
            y = {k: y_all[idx] for k, idx in sp.items()}
            cf_keys = row_keys(X["calib_final"])
            known_mask = y["test"] != H
            pre_same = int(sum(r.tobytes() in cf_keys for r in X["test"][known_mask]))
            for det in cfg["detectors"]:
                with open(run_dir / f"{hidden}__{setting}__{det}.json", encoding="utf-8") as f:
                    run = json.load(f)
                saved = np.load(run_dir / f"{hidden}__{setting}__{det}.npz")
                T_cf, T_te = statistics(det, run, cfg, known, X, y, N)
                tk = T_te[known_mask]
                rows.append({
                    "hidden": hidden, "setting": setting, "detector": det,
                    "T matches saved (max abs diff)": float(np.abs(T_te - saved["T"]).max()),
                    "(a) known test rows identical to a calib_final row: raw features": raw_same,
                    "(a) ... after preprocessing": pre_same,
                    "n_final": len(T_cf), "(b) distinct T in calib_final": len(np.unique(T_cf)),
                    "(b) known test rows with T equal to a calib_final T": int(np.isin(tk, T_cf).sum()),
                    "n known test rows": int(known_mask.sum()),
                    **{f"(c) rate alpha={a}": run["metrics"][f"hybrid|alpha={a}"]["false_unknown_rate_known"]
                       for a in cfg["calibration"]["alphas"]},
                })
                print({k: v for k, v in rows[-1].items() if not k.startswith("(c)")})
    out = ROOT / "outputs" / "tables" / f"calibration_excess_checks_{Path(cfg['config_path']).stem}.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")
    print("Saved", out)


if __name__ == "__main__":
    main()
