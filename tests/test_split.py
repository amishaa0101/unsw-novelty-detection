"""Tests for the leave-one-attack-out split."""
import numpy as np
import pandas as pd
import pytest

from src.split import make_splits

SPLIT_CFG = {"train": 0.6, "calib_norm": 0.1, "calib_final": 0.1, "test": 0.2}
SIZES = {"Normal": 5000, "Generic": 2000, "DoS": 800, "Worms": 150}


@pytest.fixture
def df():
    labels = np.concatenate([[name] * n for name, n in SIZES.items()])
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.normal(size=len(labels)), "attack_cat": labels})


def test_hidden_class_only_in_test(df):
    sp = make_splits(df, hidden="DoS", split_cfg=SPLIT_CFG, seed=42)
    for part in ["train", "calib_norm", "calib_final"]:
        assert (df.loc[sp[part], "attack_cat"] == "DoS").sum() == 0
    # All hidden rows end up in test.
    assert (df.loc[sp["test"], "attack_cat"] == "DoS").sum() == SIZES["DoS"]


def test_splits_disjoint_and_complete(df):
    sp = make_splits(df, hidden="Worms", split_cfg=SPLIT_CFG, seed=42)
    all_idx = np.concatenate(list(sp.values()))
    assert len(all_idx) == len(set(all_idx)), "a row appears in two splits"
    assert set(all_idx) == set(df.index), "some rows are in no split"


@pytest.mark.parametrize("hidden", [None, "Generic"])
def test_per_class_proportions(df, hidden):
    sp = make_splits(df, hidden=hidden, split_cfg=SPLIT_CFG, seed=42)
    for cls, n in SIZES.items():
        if cls == hidden:
            continue
        for part, frac in SPLIT_CFG.items():
            got = (df.loc[sp[part], "attack_cat"] == cls).sum() / n
            # Rounding in stratified splits moves at most a couple of rows per class.
            assert abs(got - frac) <= 0.01 + 2 / n, f"{cls} in {part}: {got:.3f} vs {frac}"


def test_same_seed_same_split(df):
    a = make_splits(df, hidden="DoS", split_cfg=SPLIT_CFG, seed=7)
    b = make_splits(df, hidden="DoS", split_cfg=SPLIT_CFG, seed=7)
    for part in a:
        assert np.array_equal(a[part], b[part])
