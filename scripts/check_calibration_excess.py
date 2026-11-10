"""Calibration diagnostics. Facts only; changes nothing in the pipeline or the saved runs.

For each round of a finished run it recomputes the split, the preprocessing and the chosen detectors
(deterministic: same seed; for GMM the reg_covar the run chose), checks that the recomputed test
statistics T equal the saved ones, and reports:
  (a) known test rows whose feature vector is identical to a calib_final row
      (raw feature columns, and after preprocessing, where dropped columns can make rows identical)
  (b) ties: share of test T values exactly equal to some calib_final T value, and the dtypes of the
      detector scores and of T for calib_final and test
With --float64 it also (d) recomputes PCA's T for one round with float64 preprocessing output
(the pipeline uses float32) and reports the false unknown rates next to the saved ones.

Usage:
  python scripts/check_calibration_excess.py --config configs/default.yaml --detectors pca knn [--float64]
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
from src.preprocess import OTHER, Preprocessor  # noqa: E402
from src.split import make_splits  # noqa: E402


def row_keys(A):
    return {r.tobytes() for r in np.ascontiguousarray(A)}


def transform_float64(prep, df):
    """Same as Preprocessor.transform but keeps float64 (diagnostic only; the pipeline uses float32)."""
    num = (prep._log(df[prep.num_cols].to_numpy(dtype=np.float64)) - prep.mean_) / prep.std_
    parts = [num]
    for c in prep.text_cols:
        cats = prep.categories_[c]
        values = df[c].where(df[c].isin(cats), OTHER).to_numpy()
        parts.append((values[:, None] == np.array(cats)[None, :]).astype(np.float64))
    return np.hstack(parts)


def statistics(det, run, cfg, known, X, y, N):
    """Recompute T for calib_final and test exactly as the experiment did. Also returns dtypes."""
    seed, ccfg = cfg["seed"], cfg["calibration"]
    if det == "baseline":
        clf = ConfidenceBaseline(seed).fit(X["train"], y["train"])
        s_cf, s_te = clf.predict(X["calib_final"], N)[0], clf.predict(X["test"], N)[0]
        return s_cf, s_te, {"score dtype": str(s_cf.dtype)}
    if det == "gmm":
        g = cfg["gmm"]
        make = lambda: GMMDetector(g["k_grid"], g["covariance_type"], g["n_init"], run["reg_covar_chosen"],
                                   g["max_fit_rows"], seed)
    elif det == "pca":
        make = lambda: PCADetector(cfg["pca"]["explained_variance"])
    else:
        make = lambda: KNNDetector(cfg["knn"]["k"])
    pcd = PerClassDetector(make, known).fit(X["train"], y["train"])
    S = {k: pcd.score(X[k]) for k in ["calib_norm", "calib_final", "test"]}
    if det in ("pca", "knn"):
        S = {k: np.log(v + ccfg["log_eps"]) for k, v in S.items()}
    norm = cal.fit_normaliser(S["calib_norm"], y["calib_norm"], known, ccfg["min_scale"])
    T_cf = cal.combine(cal.normalise(S["calib_final"], norm), known)[0]
    T_te = cal.combine(cal.normalise(S["test"], norm), known)[0]
    return T_cf, T_te, {"input dtype": str(X["test"].dtype),
                        "score dtype calib_final": str(S["calib_final"].dtype), "score dtype test": str(S["test"].dtype),
                        "T dtype calib_final": str(T_cf.dtype), "T dtype test": str(T_te.dtype)}


def rates(T_cf, T_te, known_mask, alphas):
    P = cal.conformal_pvalues(T_cf, T_te)
    return {a: float((P[known_mask] <= a).mean()) for a in alphas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--hidden", nargs="*", help="hidden classes to check (default: all)")
    ap.add_argument("--detectors", nargs="*", help="detectors to check (default: all in the config)")
    ap.add_argument("--float64", action="store_true", help="also run check (d) for PCA on one round")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.detectors:
        cfg["detectors"] = args.detectors
    name = Path(cfg["config_path"]).stem
    run_dir = ROOT / "outputs" / "runs" / name
    df = pd.read_parquet(ROOT / cfg["processed_dir"] / "clean.parquet").reset_index(drop=True)
    names = class_names(df)
    code = {n: i for i, n in enumerate(names)}
    y_all = df["attack_cat"].map(code).to_numpy()
    feats = feature_columns(df)
    alphas = cfg["calibration"]["alphas"]

    # Round for check (d): the PCA round with the largest pooled false unknown rate at the main alpha.
    d_round = None
    if args.float64:
        best = None
        for p in run_dir.glob("*__pca.json"):
            r = json.load(open(p, encoding="utf-8"))
            v = r["metrics"][f"hybrid|alpha={cfg['calibration']['main_alpha']}"]["false_unknown_rate_known"]
            if best is None or v > best[0]:
                best = (v, r["hidden"], r["setting"])
        d_round = best[1:]
        print("check (d) round:", d_round)

    rows, d_rows = [], []
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
            known_mask = y["test"] != H
            cf_keys = row_keys(X["calib_final"])
            pre_same = int(sum(r.tobytes() in cf_keys for r in X["test"][known_mask]))
            for det in cfg["detectors"]:
                with open(run_dir / f"{hidden}__{setting}__{det}.json", encoding="utf-8") as f:
                    run = json.load(f)
                saved = np.load(run_dir / f"{hidden}__{setting}__{det}.npz")
                T_cf, T_te, dtypes = statistics(det, run, cfg, known, X, y, N)
                tied = np.isin(T_te, T_cf)
                rows.append({
                    "seed": cfg["seed"], "hidden": hidden, "setting": setting, "detector": det,
                    "T matches saved (max abs diff)": float(np.abs(T_te - saved["T"]).max()),
                    "(a) known test rows identical to a calib_final row, raw features": raw_same,
                    "(a) same, after preprocessing": pre_same,
                    "n_final": len(T_cf), "distinct T in calib_final": len(np.unique(T_cf)),
                    "(b) test rows with T equal to a calib_final T": int(tied.sum()),
                    "(b) share of test rows": float(tied.mean()),
                    "(b) share of known test rows": float(tied[known_mask].mean()),
                    **dtypes,
                })
                print({k: v for k, v in rows[-1].items() if not k.startswith("dtype")})

            if d_round == (hidden, setting):
                # (d) float64 preprocessing output, PCA refitted, T and p-values recomputed.
                X64 = {k: transform_float64(prep, df.loc[idx]) for k, idx in sp.items()}
                T_cf64, T_te64, dt64 = statistics("pca", None, cfg, known, X64, y, N)
                with open(run_dir / f"{hidden}__{setting}__pca.json", encoding="utf-8") as f:
                    run = json.load(f)
                r64 = rates(T_cf64, T_te64, known_mask, alphas)
                for a in alphas:
                    d_rows.append({"seed": cfg["seed"], "hidden": hidden, "setting": setting, "alpha": a,
                                   "rate saved (float32 input)": run["metrics"][f"hybrid|alpha={a}"]["false_unknown_rate_known"],
                                   "rate recomputed (float64 input)": r64[a],
                                   "input dtype": dt64["input dtype"], "T dtype": dt64["T dtype test"],
                                   "max abs diff in T (test)": float(np.abs(T_te64 - np.load(run_dir / f"{hidden}__{setting}__pca.npz")["T"]).max())})
                print(d_rows[-len(alphas):])

    out_dir = ROOT / "outputs" / "tables"
    pd.DataFrame(rows).to_csv(out_dir / f"calibration_checks_{name}.csv", index=False, float_format="%.6g")
    if d_rows:
        pd.DataFrame(d_rows).to_csv(out_dir / f"calibration_float64_{name}.csv", index=False, float_format="%.6g")
    print("Saved tables for", name)


if __name__ == "__main__":
    main()
