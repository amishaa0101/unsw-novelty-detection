"""Tests for the single-feature shortcut audit."""
import numpy as np
import pandas as pd

from src.shortcut_audit import run_audit


def make_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    cat = rng.choice(["Normal", "DoS", "Worms"], size=n, p=[0.6, 0.3, 0.1])
    is_attack = (cat != "Normal").astype(int)
    return pd.DataFrame({
        "leaky": is_attack * 10 + rng.normal(scale=0.1, size=n),  # gives the label away
        "noise": rng.normal(size=n),                               # unrelated to the label
        "service": np.where(cat == "Worms", "smtp", rng.choice(["http", "-"], size=n)),
        "attack_cat": cat,
        "label": is_attack,
    })


def test_leaky_feature_high_noise_feature_near_half():
    t = run_audit(make_df(), ["DoS", "Worms"], seed=42).set_index("feature")
    assert t.loc["leaky", "auroc_overall"] > 0.99
    assert t.loc["leaky", "suspect_overall"]
    assert abs(t.loc["noise", "auroc_overall"] - 0.5) < 0.1
    assert not t.loc["noise", "suspect_any_attack"]


def test_per_attack_column_catches_single_attack_leak():
    # 'service' gives away Worms only, so overall AUROC is modest but the Worms column is ~1.
    t = run_audit(make_df(), ["DoS", "Worms"], seed=42).set_index("feature")
    assert t.loc["service", "auroc_Worms"] > 0.99
    assert t.loc["service", "auroc_overall"] < 0.95
    assert t.loc["service", "suspect_any_attack"]


def test_labels_never_audited_as_features():
    t = run_audit(make_df(), ["DoS", "Worms"], seed=42)
    assert not {"label", "attack_cat"} & set(t["feature"])
