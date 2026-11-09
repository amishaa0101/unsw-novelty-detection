"""One leave-one-attack-out round: split, preprocess, fit, calibrate, decide, measure, save."""
import json
import logging
import time

import numpy as np

from src import calibration as cal
from src.baseline import ConfidenceBaseline
from src.detectors import KNNDetector, PCADetector, PerClassDetector, select_gmm_reg_covar
from src.metrics import auroc_hidden_vs_normal, bootstrap, false_unknown_by_class, outcome_metrics
from src.preprocess import Preprocessor
from src.split import make_splits

log = logging.getLogger("experiment")


def class_names(df):
    """Fixed order of class codes: Normal = 0, then attacks alphabetically."""
    return ["Normal"] + sorted(c for c in df["attack_cat"].unique() if c != "Normal")


def check_split(df, sp, hidden):
    """Fail loudly if the hidden class leaked into fitting/calibration or a row is in two splits."""
    for part in ["train", "calib_norm", "calib_final"]:
        n = int((df.loc[sp[part], "attack_cat"] == hidden).sum())
        assert n == 0, f"hidden class {hidden} has {n} rows in {part}"
    all_idx = np.concatenate(list(sp.values()))
    assert len(all_idx) == len(np.unique(all_idx)), "a row appears in two splits"


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def nll_summary(pcd, X, y, onehot_mask, hidden, names, n_rows, seed):
    """Share of the GMM squared-distance part that comes from one-hot columns (per class detector).

    For each known-class detector c: rows of class c (own) and rows of the hidden class,
    up to n_rows each, from the test split. Also 5 example hidden rows under the Normal detector.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for c, det in pcd.detectors.items():
        entry = {}
        for group, mask in [("own_class", y == c), ("hidden", y == hidden)]:
            idx = np.nonzero(mask)[0]
            idx = rng.choice(idx, size=min(n_rows, len(idx)), replace=False)
            b = det.nll_breakdown(X[idx].astype(np.float64), onehot_mask)
            quad = b["quad_onehot"] + b["quad_numeric"]
            entry[group] = {
                "n": int(len(idx)),
                "median_onehot_share_of_quad": float(np.median(b["quad_onehot"] / np.maximum(quad, 1e-300))),
                "median_quad_onehot": float(np.median(b["quad_onehot"])),
                "median_quad_numeric": float(np.median(b["quad_numeric"])),
                "logdet_onehot": float(b["logdet_onehot"][0]) if len(idx) else None,
                "logdet_numeric": float(b["logdet_numeric"][0]) if len(idx) else None,
            }
        out[names[c]] = entry
    idx = rng.choice(np.nonzero(y == hidden)[0], size=min(5, int((y == hidden).sum())), replace=False)
    b = pcd.detectors[0].nll_breakdown(X[idx].astype(np.float64), onehot_mask)  # code 0 = Normal
    out["examples_hidden_under_Normal"] = [{k: float(v[i]) for k, v in b.items()} for i in range(len(idx))]
    return out


def run_round(df, hidden, setting, cfg, out_dir):
    """Run every detector for one (hidden class, feature setting). Returns summary rows."""
    seed = cfg["seed"]
    assert (df.index == np.arange(len(df))).all(), "df must have a 0..n-1 index"
    names = class_names(df)
    code = {n: i for i, n in enumerate(names)}
    H, N = code[hidden], code["Normal"]
    known = [code[n] for n in names if n != hidden]
    y_all = df["attack_cat"].map(code).to_numpy()

    # 1-2. Split and check.
    sp = make_splits(df, hidden, cfg["split"], seed)
    check_split(df, sp, hidden)

    # 3. A fresh preprocessor, fitted on train only.
    prep = Preprocessor(setting, cfg["shortcut_features"], cfg["noise_features"], cfg["strict_extra_features"],
                        cfg["preprocess"]["proto_top_n"], cfg["preprocess"]["log1p"]).fit(df.loc[sp["train"]])
    assert set(prep.fit_index_) == set(sp["train"]), "preprocessor was not fitted on exactly the train rows"
    X = {k: prep.transform(df.loc[idx]) for k, idx in sp.items()}
    y = {k: y_all[idx] for k, idx in sp.items()}
    sizes = {k: {names[c]: int((v == c).sum()) for c in np.unique(v)} for k, v in y.items()}

    ccfg = cfg["calibration"]
    rows = []
    for det in cfg["detectors"]:
        extra = {}
        t0 = time.perf_counter()
        if det == "baseline":
            model = ConfidenceBaseline(seed).fit(X["train"], y["train"])
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            T_cf, _ = model.predict(X["calib_final"])  # same calib_final rows as the detectors
            T_te, best = model.predict(X["test"])
            score_s = time.perf_counter() - t0
            extra["detector_info"] = model.info()
        else:
            # 4. Fresh per-class detectors, fitted on train rows of each known class.
            if det == "gmm":
                pcd, reg, table = select_gmm_reg_covar(cfg["gmm"], seed, known, X["train"], y["train"],
                                                       X["calib_norm"], y["calib_norm"])
                extra["reg_covar_chosen"] = reg
                extra["reg_covar_table"] = {str(r): {"mean_auroc": t["mean_auroc"],
                                                     "per_class": {names[c]: a for c, a in t["per_class"].items()}}
                                            for r, t in table.items()}
                log.info("  gmm reg_covar chosen %s; mean AUROC by value: %s", reg,
                         {r: round(t["mean_auroc"], 4) for r, t in table.items()})
            elif det == "pca":
                pcd = PerClassDetector(lambda: PCADetector(cfg["pca"]["explained_variance"]), known).fit(X["train"], y["train"])
            elif det == "knn":
                pcd = PerClassDetector(lambda: KNNDetector(cfg["knn"]["k"]), known).fit(X["train"], y["train"])
            else:
                raise ValueError(f"unknown detector {det}")
            fit_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            S = {k: pcd.score(X[k]) for k in ["calib_norm", "calib_final", "test"]}
            if det in ("pca", "knn"):  # positive and skewed scores -> log
                S = {k: np.log(v + ccfg["log_eps"]) for k, v in S.items()}
            norm = cal.fit_normaliser(S["calib_norm"], y["calib_norm"], known, ccfg["min_scale"])
            assert norm["n_rows"] == len(sp["calib_norm"]), "normalisation must use calib_norm rows only"
            T_cf, _ = cal.combine(cal.normalise(S["calib_final"], norm), known)
            T_te, best = cal.combine(cal.normalise(S["test"], norm), known)
            score_s = time.perf_counter() - t0
            if norm["fallback"]:
                log.info("  %s normaliser fallback: %s", det, {names[c]: f for c, f in norm["fallback"].items()})

            extra["detector_info"] = {names[c]: i for c, i in pcd.info().items()}
            extra["normaliser"] = {"median": {names[c]: m for c, m in zip(known, norm["median"])},
                                   "scale": {names[c]: d for c, d in zip(known, norm["scale"])},
                                   "fallback": {names[c]: f for c, f in norm["fallback"].items()}}
            if det == "gmm":
                extra["nll_breakdown"] = nll_summary(pcd, X["test"], y["test"], prep.onehot_mask, H, names,
                                                     cfg["gmm"]["nll_breakdown_rows"], seed)
                log.info("  gmm K per class: %s", {n: i["k"] for n, i in extra["detector_info"].items()})
            if det == "pca":
                log.info("  pca k per class: %s", {n: i["k"] for n, i in extra["detector_info"].items()})

        # 5. Pooled conformal p-values from calib_final, decisions at every alpha and with BH.
        P = cal.conformal_pvalues(T_cf, T_te)
        decisions = {f"alpha={a}": cal.decide(P, best, a) for a in ccfg["alphas"]}
        decisions["bh"] = np.where(cal.benjamini_hochberg(P, ccfg["bh_q"]), cal.UNKNOWN, best)

        yt = y["test"]
        point = {k: outcome_metrics(p, yt, H, N) for k, p in decisions.items()}
        by_class = {k: {names[c]: r for c, r in false_unknown_by_class(p, yt, H).items()} for k, p in decisions.items()}
        auroc = auroc_hidden_vs_normal(T_te, yt, H, N)
        t0 = time.perf_counter()
        ci = bootstrap(decisions, T_te, yt, H, N, cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["ci"], seed)
        boot_s = time.perf_counter() - t0

        record = {
            "hidden": hidden, "setting": setting, "detector": det, "seed": seed,
            "class_codes": names, "n_features": len(prep.feature_names), "feature_names": prep.feature_names,
            "split_sizes": sizes, "runtime_s": {"fit": fit_s, "score": score_s, "bootstrap": boot_s},
            "auroc_hidden_vs_normal": auroc, "auroc_ci": ci.pop("auroc"),
            "metrics": point, "metrics_ci": ci, "false_unknown_by_class": by_class,
            "n_calib_final": int(len(T_cf)), **extra, "config": cfg,
        }
        stem = f"{hidden}__{setting}__{det}"
        with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1, default=_json_default)
        np.savez_compressed(out_dir / f"{stem}.npz", P=P, T=T_te, best=best, y=yt, row=sp["test"])

        m = point[f"alpha={ccfg['main_alpha']}"]
        rows.append({"hidden": hidden, "setting": setting, "detector": det,
                     "detection_rate": m["detection_rate"], "frac_unknown": m["frac_unknown"],
                     "frac_wrong_known": m["frac_wrong_known"], "frac_missed": m["frac_missed"],
                     "false_alarm_rate": m["false_alarm_rate"], "false_unknown_rate_known": m["false_unknown_rate_known"],
                     "auroc": auroc, "fit_s": fit_s, "score_s": score_s, "bootstrap_s": boot_s})
        log.info("  %-8s det=%.3f unk=%.3f FA=%.4f FUknown=%.4f AUROC=%.3f fit=%.1fs score=%.1fs boot=%.1fs",
                 det, m["detection_rate"], m["frac_unknown"], m["false_alarm_rate"],
                 m["false_unknown_rate_known"], auroc, fit_s, score_s, boot_s)
    return rows
