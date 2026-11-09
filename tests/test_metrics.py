"""Tests for metrics and bootstrap."""
import numpy as np

from src.calibration import UNKNOWN
from src.metrics import auroc_hidden_vs_normal, bootstrap, false_unknown_by_class, outcome_metrics

NORMAL, A, HIDDEN = 0, 1, 9


def example():
    y = np.array([HIDDEN] * 4 + [NORMAL] * 4 + [A] * 2)
    pred = np.array([UNKNOWN, UNKNOWN, A, NORMAL,      # hidden: 2 unknown, 1 wrong known, 1 missed
                     NORMAL, NORMAL, NORMAL, UNKNOWN,  # normal: 1 false unknown
                     A, UNKNOWN])                      # known attack: 1 correct, 1 unknown
    return pred, y


def test_outcomes_by_hand():
    pred, y = example()
    m = outcome_metrics(pred, y, HIDDEN, NORMAL)
    assert m["frac_unknown"] == 0.5 and m["frac_wrong_known"] == 0.25 and m["frac_missed"] == 0.25
    assert m["frac_unknown"] + m["frac_wrong_known"] + m["frac_missed"] == 1.0
    assert m["detection_rate"] == 0.75
    assert m["false_alarm_rate"] == 0.25 and m["false_unknown_rate_normal"] == 0.25
    assert m["false_unknown_rate_known"] == 2 / 6
    assert m["known_attack_correct"] == 0.5 and m["known_attack_flagged_unknown"] == 0.5
    assert m["false_discovery_proportion"] == 2 / 4  # 4 flagged, 2 of them known
    assert false_unknown_by_class(pred, y, HIDDEN) == {NORMAL: 0.25, A: 0.5}


def test_auroc_perfect_and_reversed():
    y = np.array([HIDDEN, HIDDEN, NORMAL, NORMAL])
    assert auroc_hidden_vs_normal(np.array([5.0, 6, 1, 2]), y, HIDDEN, NORMAL) == 1.0
    assert auroc_hidden_vs_normal(np.array([1.0, 2, 5, 6]), y, HIDDEN, NORMAL) == 0.0


def test_bootstrap_contains_point_and_is_seeded():
    rng = np.random.default_rng(0)
    y = np.array([HIDDEN] * 300 + [NORMAL] * 600 + [A] * 200)
    pred = np.where(rng.uniform(size=len(y)) < 0.3, UNKNOWN, y)
    nov = rng.normal(size=len(y)) + (y == HIDDEN)
    b1 = bootstrap({"x": pred}, nov, y, HIDDEN, NORMAL, 200, 0.95, seed=1)
    b2 = bootstrap({"x": pred}, nov, y, HIDDEN, NORMAL, 200, 0.95, seed=1)
    assert b1 == b2
    point = outcome_metrics(pred, y, HIDDEN, NORMAL)["frac_unknown"]
    lo, hi = b1["x"]["frac_unknown"]
    assert lo <= point <= hi and hi - lo < 0.2
