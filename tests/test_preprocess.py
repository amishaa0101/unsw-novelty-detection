"""Tests for the Preprocessor."""
import numpy as np
import pandas as pd
import pytest

from src.preprocess import Preprocessor

SHORTCUT = ["sttl", "dttl", "ct_state_ttl", "is_sm_ips_ports"]
NOISE = ["stcpb", "dtcpb"]
STRICT = ["tcprtt", "synack", "ackdat"]


def make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "sbytes": rng.integers(0, 10_000, n),
        "dur": rng.exponential(1.0, n),
        "sttl": rng.choice([31, 62, 254], n),
        "dttl": rng.choice([29, 252], n),
        "ct_state_ttl": rng.integers(0, 3, n),
        "is_sm_ips_ports": rng.integers(0, 2, n),
        "stcpb": rng.integers(0, 2**32, n),
        "dtcpb": rng.integers(0, 2**32, n),
        "tcprtt": rng.exponential(0.1, n),
        "synack": rng.exponential(0.05, n),
        "ackdat": rng.exponential(0.05, n),
        "proto": rng.choice(["tcp", "udp", "arp"], n),
        "service": rng.choice(["-", "http", "dns"], n),
        "state": rng.choice(["FIN", "INT", "CON"], n),
        "attack_cat": rng.choice(["Normal", "DoS"], n),
        "source_file": "a.csv",
    })
    df["label"] = (df["attack_cat"] != "Normal").astype(int)
    return df


def prep(setting, **kw):
    return Preprocessor(setting, SHORTCUT, NOISE, STRICT, **kw)


def test_transform_does_not_change_fitted_stats():
    train, test = make_df(seed=0), make_df(seed=1)
    p = prep("with_shortcut").fit(train)
    mean, std = p.mean_.copy(), p.std_.copy()
    p.transform(test)
    assert np.array_equal(mean, p.mean_) and np.array_equal(std, p.std_)
    # Train data is standardised to mean 0, std 1 on numeric columns.
    Xn = p.transform(train)[:, ~p.onehot_mask]
    assert np.allclose(Xn.mean(axis=0), 0, atol=1e-4) and np.allclose(Xn.std(axis=0), 1, atol=1e-3)


def test_unseen_service_maps_to_other():
    train, test = make_df(seed=0), make_df(n=5, seed=1)
    test["service"] = "never-seen"
    p = prep("with_shortcut").fit(train)
    X = p.transform(test)
    col = p.feature_names.index("service=other")
    assert (X[:, col] == 1).all()


@pytest.mark.parametrize("setting, shortcut_present, strict_present", [
    ("with_shortcut", True, True), ("without_shortcut", False, True), ("strict", False, False)])
def test_feature_sets_per_setting(setting, shortcut_present, strict_present):
    p = prep(setting).fit(make_df())
    names = set(p.num_cols)
    assert all((f in names) == shortcut_present for f in SHORTCUT)
    assert all((f in names) == strict_present for f in STRICT)
    assert not names & set(NOISE), "noise features must be dropped in every setting"


def test_labels_and_source_file_never_features():
    p = prep("with_shortcut").fit(make_df())
    joined = " ".join(p.feature_names)
    for bad in ["label", "attack_cat", "source_file"]:
        assert bad not in joined


def test_source_file_as_feature_raises():
    # If a future change ever let source_file through select_features, fit must fail loudly.
    import src.preprocess as pp
    original = pp.select_features
    pp.select_features = lambda columns, *a, **k: list(columns)
    try:
        with pytest.raises(AssertionError):
            prep("with_shortcut").fit(make_df())
    finally:
        pp.select_features = original


def test_proto_top_n_and_output_type():
    p = prep("with_shortcut", proto_top_n=2).fit(make_df())
    assert len(p.categories_["proto"]) == 3  # top 2 + "other"
    X = p.transform(make_df(seed=3))
    assert X.dtype == np.float32 and X.flags["C_CONTIGUOUS"]
    assert X.shape[1] == len(p.feature_names)
