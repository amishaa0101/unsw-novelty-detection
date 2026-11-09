"""Combined detector "combo_<D>", built from saved test p-values only (no re-fitting).

unknown if min(P_D(x), P_baseline(x)) <= alpha / 2 (Bonferroni), otherwise the classifier's label.
"""
import json
from pathlib import Path

import numpy as np

from src.calibration import UNKNOWN, decide_bonferroni_min
from src.experiment import _json_default
from src.metrics import auroc_hidden_vs, bootstrap, false_unknown_by_class, outcome_metrics

DETECTORS = ("gmm", "knn", "pca")


def run_combo(run_dir):
    """Write one combo JSON per (hidden, setting, detector) into run_dir/combo/. Returns the paths."""
    run_dir = Path(run_dir)
    out_dir = run_dir / "combo"
    out_dir.mkdir(exist_ok=True)
    paths = []
    for base_json in sorted(run_dir.glob("*__baseline.json")):
        with open(base_json, encoding="utf-8") as f:
            base = json.load(f)
        zb = np.load(base_json.with_suffix(".npz"))
        cfg = base["config"]
        names = base["class_codes"]
        H, N = names.index(base["hidden"]), names.index("Normal")
        for det in DETECTORS:
            stem = base_json.stem.replace("__baseline", f"__{det}")
            zd = np.load(run_dir / f"{stem}.npz")
            # Same round -> same test rows in the same order, same labels, same classifier predictions.
            for k in ["row", "y", "clf_pred"]:
                assert np.array_equal(zd[k], zb[k]), f"{stem}: {k} differs from the baseline run"
            y, labels = zb["y"], zb["clf_pred"]
            P = np.minimum(zd["P"], zb["P"])
            decisions = {f"combo|alpha={a}": decide_bonferroni_min(zd["P"], zb["P"], labels, a)
                         for a in cfg["calibration"]["alphas"]}
            # Threshold-free: a smaller combined p-value means more unusual.
            aurocs = {"novelty_auroc_vs_known": (-P, "known"), "novelty_auroc_vs_normal": (-P, "normal")}
            point = {k: outcome_metrics(p, y, H, N) for k, p in decisions.items()}
            by_class = {k: {names[c]: r for c, r in false_unknown_by_class(p, y, H).items()}
                        for k, p in decisions.items()}
            auc = {k: auroc_hidden_vs(s, y, H, N, neg) for k, (s, neg) in aurocs.items()}
            ci = bootstrap(decisions, aurocs, y, H, N, cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["ci"],
                           cfg["seed"])
            record = {"hidden": base["hidden"], "setting": base["setting"], "detector": f"combo_{det}",
                      "seed": cfg["seed"], "class_codes": names, "split_sizes": base["split_sizes"],
                      "rule": "unknown if min(P_D, P_baseline) <= alpha/2, else classifier label",
                      "auroc": auc, "auroc_ci": ci.pop("auroc"), "metrics": point, "metrics_ci": ci,
                      "false_unknown_by_class": by_class, "config": cfg}
            path = out_dir / f"{base['hidden']}__{base['setting']}__combo_{det}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=1, default=_json_default)
            paths.append(path)
    return paths
