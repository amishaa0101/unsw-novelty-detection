"""Two-stage calibration: per-class normalisation on calib_norm, pooled conformal p-value on calib_final."""
import numpy as np

UNKNOWN = -1  # code for "unknown" in prediction arrays (class codes are 0, 1, 2, ...)


def fit_normaliser(S_norm, y_norm, class_codes, min_scale=1e-6):
    """Median and IQR of each class's own scores, using calib_norm rows only.

    S_norm: (n, n_classes) scores of calib_norm rows; column i belongs to class_codes[i].
    If the IQR is below min_scale, fall back to the standard deviation, then to 1.0.
    """
    median, scale, fallback = [], [], {}
    for i, c in enumerate(class_codes):
        s = S_norm[y_norm == c, i]
        assert len(s) > 0, f"class {c} has no calib_norm rows"
        q25, q50, q75 = np.percentile(s, [25, 50, 75])
        if q75 - q25 >= min_scale:
            d = q75 - q25
        elif s.std() >= min_scale:
            d, fallback[int(c)] = s.std(), "std"
        else:
            d, fallback[int(c)] = 1.0, "one"
        median.append(q50)
        scale.append(d)
    return {"median": np.array(median), "scale": np.array(scale), "fallback": fallback, "n_rows": len(S_norm)}


def normalise(S, norm):
    """z_c(x) = (s_c(x) - m_c) / d_c, column by column."""
    return (S - norm["median"]) / norm["scale"]


def combine(Z, class_codes):
    """T(x) = min_c z_c(x): how unusual x is compared with the class it fits best."""
    best = np.asarray(class_codes)[Z.argmin(axis=1)]
    return Z.min(axis=1), best


def conformal_pvalues(calib_stats, test_stats):
    """P(x) = (1 + #{i : T_i >= T(x)}) / (n + 1), using sorting and searchsorted."""
    srt = np.sort(np.asarray(calib_stats))
    n = len(srt)
    n_ge = n - np.searchsorted(srt, test_stats, side="left")  # calibration values >= each test value
    return (1 + n_ge) / (n + 1)


def decide(P, best, alpha):
    """unknown if P(x) <= alpha, else the best-fitting known class."""
    return np.where(P <= alpha, UNKNOWN, best)


def decide_bonferroni_min(P1, P2, labels, alpha):
    """unknown if min(P1, P2) <= alpha / 2, else the given label.

    Bonferroni: P(min <= alpha/2) <= P(P1 <= alpha/2) + P(P2 <= alpha/2) <= alpha for a known flow,
    with no assumption about how the two p-values depend on each other.
    """
    return np.where(np.minimum(P1, P2) <= alpha / 2, UNKNOWN, labels)


def benjamini_hochberg(p, q):
    """Return a boolean mask of rejected hypotheses (flagged as unknown) at FDR level q."""
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p, kind="stable")
    below = p[order] <= q * np.arange(1, m + 1) / m
    reject = np.zeros(m, dtype=bool)
    if below.any():
        k = np.nonzero(below)[0].max()  # largest i with p_(i) <= i q / m
        reject[order[: k + 1]] = True
    return reject
