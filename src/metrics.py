"""Outcome rates, AUROC and bootstrap confidence intervals.

Labels and predictions are integer codes; UNKNOWN (-1) means "flagged unknown".
Integer arrays keep the 1000 bootstrap resamples fast.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from src.calibration import UNKNOWN


def _mean(mask):
    return float(mask.mean()) if mask.size else float("nan")


def outcome_metrics(pred, y, hidden, normal):
    """All threshold-based metrics for one set of predictions."""
    h, n = y == hidden, y == normal
    ka = ~h & ~n  # known attack rows
    ph = pred[h]
    frac_unknown = _mean(ph == UNKNOWN)
    frac_missed = _mean(ph == normal)
    flagged = pred == UNKNOWN
    return {
        "frac_unknown": frac_unknown,
        "frac_wrong_known": 1.0 - frac_unknown - frac_missed,
        "frac_missed": frac_missed,
        "detection_rate": 1.0 - frac_missed,
        "false_alarm_rate": _mean(pred[n] != normal),
        "false_unknown_rate_normal": _mean(pred[n] == UNKNOWN),
        "false_unknown_rate_known": _mean(pred[~h] == UNKNOWN),  # pooled over all known rows: the guarantee
        "known_attack_correct": _mean(pred[ka] == y[ka]),
        "known_attack_flagged_unknown": _mean(pred[ka] == UNKNOWN),
        # Among flows flagged unknown, the share that are really known classes.
        "false_discovery_proportion": float((~h[flagged]).mean()) if flagged.any() else 0.0,
        "n_flagged": int(flagged.sum()),
    }


def false_unknown_by_class(pred, y, hidden):
    """Share of each known class's test rows flagged unknown (the guarantee is only marginal)."""
    return {int(c): _mean(pred[y == c] == UNKNOWN) for c in np.unique(y) if c != hidden}


def auroc_hidden_vs(score, y, hidden, normal, negatives):
    """AUROC with hidden attack rows as positives; higher score = more unusual.

    negatives = "normal": Normal test rows only; negatives = "known": all known-class test rows.
    """
    neg = (y == normal) if negatives == "normal" else (y != hidden)
    m = (y == hidden) | neg
    return float(roc_auc_score((y[m] == hidden).astype(int), score[m]))


def confusion(pred, y, row_codes, col_codes):
    """Counts of (true class, assigned label). Columns include UNKNOWN."""
    r = {c: i for i, c in enumerate(row_codes)}
    k = {c: i for i, c in enumerate(col_codes)}
    M = np.zeros((len(row_codes), len(col_codes)), dtype=int)
    np.add.at(M, (np.array([r[v] for v in y]), np.array([k[v] for v in pred])), 1)
    return M


def bootstrap(decisions, aurocs, y, hidden, normal, n_resamples, ci, seed):
    """Percentile CIs. Resample rows with replacement within each group (hidden, Normal, known attacks).

    decisions: dict name -> prediction array.
    aurocs: dict name -> (score array, "normal" | "known").
    Returns dict name -> metric -> [low, high], plus "auroc" -> name -> [low, high].
    """
    rng = np.random.default_rng(seed)
    groups = [np.nonzero(y == hidden)[0], np.nonzero(y == normal)[0],
              np.nonzero((y != hidden) & (y != normal))[0]]
    samples = {name: [] for name in decisions}
    auc_samples = {name: [] for name in aurocs}
    for _ in range(n_resamples):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups if len(g)])
        yb = y[idx]
        for name, pred in decisions.items():
            samples[name].append(outcome_metrics(pred[idx], yb, hidden, normal))
        for name, (score, negatives) in aurocs.items():
            auc_samples[name].append(auroc_hidden_vs(score[idx], yb, hidden, normal, negatives))

    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    out = {}
    for name, rows in samples.items():
        out[name] = {k: [float(np.percentile([r[k] for r in rows], lo)), float(np.percentile([r[k] for r in rows], hi))]
                     for k in rows[0] if k != "n_flagged"}
    out["auroc"] = {name: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for name, v in auc_samples.items()}
    return out
