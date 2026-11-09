"""Tests for the three detectors and the per-class wrapper."""
import numpy as np
import pytest

from src.detectors import GMMDetector, KNNDetector, PCADetector, PerClassDetector, class_separation_auroc


def blob(n=500, seed=0):
    rng = np.random.default_rng(seed)
    # Elongated 2D cloud: big spread along x, small along y.
    return (rng.normal(size=(n, 2)) * np.array([3.0, 0.3])).astype(np.float32)


DETECTORS = {
    "gmm": lambda: GMMDetector([1, 2, 4], n_init=1, seed=0),
    "pca": lambda: PCADetector(0.9),
    "knn": lambda: KNNDetector(5),
}


@pytest.mark.parametrize("name", DETECTORS)
def test_far_point_scores_higher(name):
    det = DETECTORS[name]().fit(blob())
    s = det.score(np.array([[0.0, 0.0], [0.0, 10.0]], dtype=np.float32))
    assert s[1] > s[0]


def test_pca_zero_error_on_main_direction():
    X = blob()
    det = PCADetector(0.9).fit(X)
    assert det.k == 1
    on_axis = det.mean_ + 5 * det.V_[0]  # a point exactly on the first principal direction
    assert det.score(on_axis[None, :])[0] < 1e-10


def test_pca_k_capped():
    det = PCADetector(0.999999).fit(blob(n=3))
    assert det.k <= 1  # min(n_rows - 1, n_features - 1) = 1


def test_knn_new_index_each_fit():
    det = KNNDetector(3)
    q = np.array([[0.0, 0.0]], dtype=np.float32)
    s1 = det.fit(blob(seed=0)).score(q)
    s2 = det.fit(blob(seed=0) + 50).score(q)  # refit on shifted data must not keep old points
    assert s2[0] > 40 and s1[0] < 1


def test_knn_distance_is_not_squared():
    det = KNNDetector(1).fit(np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32))
    assert det.score(np.array([[3.0, 4.0]], dtype=np.float32))[0] == pytest.approx(5.0)


def test_k_cap_rule_a():
    from src.detectors import k_cap
    assert k_cap(95, 69) == 1        # Worms-sized class: (96) // 139 = 0 -> minimum 1
    assert k_cap(868, 69) == 6       # Shellcode-sized: 869 // 139
    assert k_cap(20000, 69) == 143   # big class: cap far above the K grid
    assert k_cap(0, 10) == 1


def test_gmm_k_capped_by_rows():
    # 2D data: 2d + 1 = 5 parameters per component. 24 rows -> cap 25 // 5 = 5, so K in {1, 2, 4}.
    det = GMMDetector([1, 2, 4, 8, 16], n_init=1, seed=0).fit(blob(n=24))
    assert det.k_max == 5 and set(det.bic) == {1, 2, 4} and det.k <= 5
    det = GMMDetector([1, 2, 4], n_init=1, seed=0).fit(blob(n=8))  # 9 // 5 = 1
    assert set(det.bic) == {1}


def test_gmm_nll_breakdown_adds_up_for_one_component():
    X = blob()
    det = GMMDetector([1], n_init=1, seed=0).fit(X)
    b = det.nll_breakdown(X[:20].astype(np.float64), np.array([False, True]))
    total = b["quad_onehot"] + b["quad_numeric"] + b["logdet_onehot"] + b["logdet_numeric"]
    assert np.allclose(total, det.score(X[:20]), rtol=1e-6)


def test_gmm_component_sizes_sum_to_rows():
    det = GMMDetector([1, 2, 4], n_init=1, seed=0).fit(blob(n=300))
    assert sum(det.component_sizes) == 300 and len(det.component_sizes) == det.k


def test_parallel_gmm_selection_equals_sequential():
    from src.detectors import select_gmm_reg_covar
    X = np.vstack([blob(n=200, seed=1), blob(n=100, seed=2) + 10]).astype(np.float32)
    y = np.array([0] * 200 + [1] * 100)
    cfg = {"reg_covar_grid": [1e-3, 1e-1], "k_grid": [1, 2], "covariance_type": "diag", "n_init": 1, "max_fit_rows": 10000}
    a, reg_a, _ = select_gmm_reg_covar(cfg, 0, [0, 1], X, y, X, y, n_jobs=1)
    b, reg_b, _ = select_gmm_reg_covar(cfg, 0, [0, 1], X, y, X, y, n_jobs=2)
    assert reg_a == reg_b and np.allclose(a.score(X), b.score(X))


def test_per_class_wrapper_shape_and_separation():
    X = np.vstack([blob(seed=1), blob(seed=2) + np.array([0, 20], dtype=np.float32)])
    y = np.array(["A"] * 500 + ["B"] * 500)
    det = PerClassDetector(lambda: KNNDetector(5), ["A", "B"]).fit(X, y)
    S = det.score(X)
    assert S.shape == (1000, 2)
    auc = class_separation_auroc(S, y, ["A", "B"])
    assert auc["A"] > 0.99 and auc["B"] > 0.99
