"""Tests for the drift-shift simulators and the retrain verdict in src/drift.py.

The simulators are what make the monitoring demo meaningful, so we check each
one actually produces the distribution change it claims, and that the verdict
logic trips on the right signals.
"""

import numpy as np
import pandas as pd

from src.drift import (
    shift_age_older,
    shift_none,
    shift_pipeline_break,
    shift_prevalence_surge,
    verdict,
)


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age_midpoint": rng.choice([15, 35, 55, 75, 95], n),
            "discharge_disposition_id": rng.choice(["1", "2", "3", "6"], n),
            "readmitted_lt30": rng.integers(0, 2, n),
        }
    )


def test_shift_none_is_identity():
    df = _toy()
    out = shift_none(df, np.random.default_rng(1))
    pd.testing.assert_frame_equal(df.reset_index(drop=True),
                                  out.reset_index(drop=True))


def test_age_shift_skews_older():
    df = _toy()
    out = shift_age_older(df, np.random.default_rng(1))
    assert out["age_midpoint"].mean() > df["age_midpoint"].mean() + 5
    assert len(out) == len(df)


def test_pipeline_break_collapses_field():
    df = _toy()
    out = shift_pipeline_break(df, np.random.default_rng(1))
    assert out["discharge_disposition_id"].nunique() == 1
    assert set(out["discharge_disposition_id"]) == {"1"}


def test_prevalence_surge_raises_positive_rate():
    df = _toy()
    out = shift_prevalence_surge(df, np.random.default_rng(1))
    assert out["readmitted_lt30"].mean() > df["readmitted_lt30"].mean() + 0.05


def test_verdict_trips_on_performance_and_drift():
    ref = {"auroc": 0.66, "brier": 0.076}

    # Healthy current window: no alert.
    good = {"auroc": 0.66, "brier": 0.077}
    v = verdict(good, ref, drift_share=0.0)
    assert v["retrain_recommended"] is False

    # AUROC collapse trips the alert.
    bad_auroc = {"auroc": 0.60, "brier": 0.077}
    v = verdict(bad_auroc, ref, drift_share=0.0)
    assert v["retrain_recommended"] is True
    assert any("AUROC" in r for r in v["reasons"])

    # Calibration blow-up trips the alert.
    bad_brier = {"auroc": 0.66, "brier": 0.18}
    v = verdict(bad_brier, ref, drift_share=0.0)
    assert v["retrain_recommended"] is True
    assert any("Brier" in r for r in v["reasons"])

    # Widespread data drift trips the alert on its own.
    v = verdict(good, ref, drift_share=0.5)
    assert v["retrain_recommended"] is True
    assert any("drift" in r for r in v["reasons"])
