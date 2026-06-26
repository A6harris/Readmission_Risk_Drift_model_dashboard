"""Tests for the subgroup-metric logic in src/fairness.py.

Uses a small constructed frame with a known disparity so the metric table and
disparity gaps are checkable by hand.
"""

import numpy as np
import pandas as pd

from src.fairness import _disparities, subgroup_metrics


def _toy():
    # Two groups. Group A: model is perfect. Group B: model always says 0,
    # so its TPR is 0 -> a large, known TPR gap. Each group has > 100 rows so
    # neither is flagged low-evidence.
    n = 150
    y_a = np.array([1, 0] * (n // 2))
    p_a = np.where(y_a == 1, 0.9, 0.1)  # perfectly separating scores
    y_b = np.array([1, 0] * (n // 2))
    p_b = np.full(n, 0.01)  # always below threshold -> never flags

    y = np.concatenate([y_a, y_b])
    proba = np.concatenate([p_a, p_b])
    sensitive = pd.Series(["A"] * n + ["B"] * n)
    return y, proba, sensitive


def test_subgroup_metrics_table():
    y, proba, sensitive = _toy()
    y_pred = (proba >= 0.5).astype(int)
    table = subgroup_metrics(y, proba=proba, y_pred=y_pred, sensitive=sensitive)

    assert set(table.index) == {"A", "B"}
    # Group A is perfectly separated at threshold 0.5.
    assert table.loc["A", "tpr"] == 1.0
    assert table.loc["A", "fpr"] == 0.0
    assert table.loc["A", "auroc"] == 1.0
    # Group B never flags anyone -> zero recall, zero selection.
    assert table.loc["B", "tpr"] == 0.0
    assert table.loc["B", "selection_rate"] == 0.0
    # Both groups are large enough to be reliable.
    assert not table["low_evidence"].any()


def test_disparities_pick_extremes():
    y, proba, sensitive = _toy()
    y_pred = (proba >= 0.5).astype(int)
    table = subgroup_metrics(y, proba=proba, y_pred=y_pred, sensitive=sensitive)
    disp = _disparities(table)

    assert disp["tpr_gap"] == 1.0
    assert disp["tpr_min_group"] == "B"
    assert disp["tpr_max_group"] == "A"


def test_low_evidence_excluded_from_disparities():
    # A tiny third group (n=10) must be ignored when computing gaps.
    y, proba, sensitive = _toy()
    y = np.concatenate([y, [1, 0] * 5])
    proba = np.concatenate([proba, np.full(10, 0.99)])
    sensitive = pd.concat([sensitive, pd.Series(["C"] * 10)], ignore_index=True)
    y_pred = (proba >= 0.5).astype(int)

    table = subgroup_metrics(y, proba=proba, y_pred=y_pred, sensitive=sensitive)
    assert bool(table.loc["C", "low_evidence"])
    disp = _disparities(table)
    # C (perfectly separating, high selection) must not widen the gaps.
    assert disp["selection_rate_max_group"] in {"A", "B"}
