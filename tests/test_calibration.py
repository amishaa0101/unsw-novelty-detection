"""Tests for the two-stage calibration."""
import numpy as np
from scipy.stats import false_discovery_control

from src.calibration import (UNKNOWN, benjamini_hochberg, combine, conformal_pvalues, decide,
                             fit_normaliser, normalise)


def test_pvalue_hand_example():
    # calib [1,2,3,4], test 3.5: one calib value >= 3.5, so P = (1 + 1) / 5 = 0.4
    assert conformal_pvalues([1, 2, 3, 4], np.array([3.5]))[0] == 0.4
    # ties count as ">=": test 3 -> values 3 and 4 -> (1 + 2) / 5
    assert conformal_pvalues([1, 2, 3, 4], np.array([3.0]))[0] == 0.6


def test_pvalue_floor():
    calib = np.random.default_rng(0).normal(size=99)
    P = conformal_pvalues(calib, np.array([1e9, -1e9, 0.0]))
    assert P.min() >= 1 / (99 + 1)
    assert P[0] == 1 / 100 and P[1] == 1.0


def synthetic_scores(rng, sizes, centers, scales):
    """Scores matrix where class c's own column is N(center_c, scale_c) and others are larger."""
    y = np.concatenate([[c] * n for c, n in enumerate(sizes)])
    S = np.empty((len(y), len(sizes)))
    for c in range(len(sizes)):
        own = y == c
        S[own, c] = rng.normal(centers[c], scales[c], own.sum())
        S[~own, c] = rng.normal(centers[c] + 6 * scales[c], scales[c], (~own).sum())
    return S, y


def test_pooled_false_unknown_rate_close_to_alpha():
    # Classes of very different sizes and scales: guarantee is marginal over the known mix.
    rng = np.random.default_rng(1)
    sizes, centers, scales = [20000, 3000, 300], [0.0, 50.0, -10.0], [1.0, 20.0, 0.1]
    codes = [0, 1, 2]
    rates = {0.01: [], 0.05: []}
    for _ in range(20):
        S_n, y_n = synthetic_scores(rng, sizes, centers, scales)
        S_f, y_f = synthetic_scores(rng, sizes, centers, scales)
        S_t, y_t = synthetic_scores(rng, sizes, centers, scales)
        norm = fit_normaliser(S_n, y_n, codes)
        T_f, _ = combine(normalise(S_f, norm), codes)
        T_t, best = combine(normalise(S_t, norm), codes)
        P = conformal_pvalues(T_f, T_t)
        for a in rates:
            rates[a].append((decide(P, best, a) == UNKNOWN).mean())
    for a, r in rates.items():
        # Mean over 20 repeats of ~23k rows each; allow a small sampling error.
        assert abs(np.mean(r) - a) < 0.25 * a + 0.002, (a, np.mean(r))


def test_normaliser_uses_only_calib_norm():
    rng = np.random.default_rng(2)
    S_n, y_n = synthetic_scores(rng, [500, 200], [0, 5], [1, 2])
    norm1 = fit_normaliser(S_n, y_n, [0, 1])
    # The function only receives calib_norm arrays; changing anything else cannot change it,
    # and the stored row count matches calib_norm.
    norm2 = fit_normaliser(S_n.copy(), y_n.copy(), [0, 1])
    assert np.array_equal(norm1["median"], norm2["median"]) and np.array_equal(norm1["scale"], norm2["scale"])
    assert norm1["n_rows"] == len(S_n)
    own = S_n[y_n == 0, 0]
    assert norm1["median"][0] == np.median(own)


def test_iqr_fallback_for_constant_class():
    S = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0], [2.0, 2.0]])
    y = np.array([0, 0, 0, 1])
    norm = fit_normaliser(S, y, [0, 1])
    assert norm["scale"][0] == 1.0 and norm["fallback"][0] == "one"
    S2 = np.array([[1.0], [1.0], [1.0], [1.0], [5.0]])
    norm2 = fit_normaliser(S2, np.zeros(5, dtype=int), [0])
    assert norm2["fallback"][0] == "std"


def test_bh_matches_scipy():
    rng = np.random.default_rng(3)
    for _ in range(50):
        p = np.concatenate([rng.uniform(size=200), rng.uniform(0, 0.001, size=rng.integers(0, 30))])
        for q in [0.01, 0.05, 0.1]:
            ours = benjamini_hochberg(p, q)
            ref = false_discovery_control(p, method="bh") <= q
            assert np.array_equal(ours, ref)


def test_bonferroni_min_controls_rate():
    from src.calibration import decide_bonferroni_min
    rng = np.random.default_rng(4)
    n = 200_000
    P1 = rng.uniform(size=n)
    P2 = np.clip(P1 + rng.normal(0, 0.3, n), 0, 1)  # dependent on P1, still roughly uniform-ish
    P2 = (np.argsort(np.argsort(P2)) + 1) / n          # make P2 exactly uniform by ranks
    for a in [0.01, 0.05, 0.1]:
        rate = (decide_bonferroni_min(P1, P2, np.zeros(n, dtype=int), a) == UNKNOWN).mean()
        assert rate <= a + 3 * np.sqrt(a / n), (a, rate)
    assert decide_bonferroni_min(np.array([0.004, 0.5]), np.array([0.9, 0.006]), np.array([1, 2]), 0.01).tolist() == [UNKNOWN, 2]


def test_decide():
    P = np.array([0.001, 0.5, 0.01])
    assert decide(P, np.array([3, 4, 5]), 0.01).tolist() == [UNKNOWN, 4, UNKNOWN]
