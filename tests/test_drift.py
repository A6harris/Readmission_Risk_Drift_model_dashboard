"""Tests for the drift-shift simulators and the retrain verdict in src/drift.py.

The simulators are what make the monitoring demo meaningful, so we check each
one actually produces the distribution change it claims, and that the verdict
logic trips on the right signals.
"""

import numpy as np
import pandas as pd

from src.drift import (
    _extract_drifted_columns,
    plot_drift_panel,
    plot_drift_timeline,
    scenario_verdict,
    shift_age_older,
    shift_none,
    shift_pipeline_break,
    shift_prevalence_surge,
    verdict,
    window_status,
)


def _toy_summary():
    """A minimal drift summary covering both verdicts, for the figure test."""
    thresholds = {"drift_share_alert": 0.30, "auroc_drop_alert": 0.03,
                  "brier_rise_alert": 0.02, "outreach_threshold": 0.10}
    ref = {"n": 1000, "auroc": 0.66, "brier": 0.076, "prevalence": 0.09}

    def sc(drift, auroc, brier, retrain):
        return {"drift_share": drift, "auroc": auroc, "brier": brier,
                "prevalence": 0.09, "retrain_recommended": retrain}

    return {
        "reference": ref,
        "thresholds": thresholds,
        "scenarios": {
            "baseline": sc(0.0, 0.66, 0.077, False),
            "age_shift": sc(0.07, 0.66, 0.090, False),
            "pipeline_break": sc(0.02, 0.61, 0.084, True),
            "prevalence_surge": sc(0.0, 0.67, 0.184, True),
        },
    }


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


def test_plot_drift_panel_writes_a_figure(tmp_path):
    out = tmp_path / "drift_panel.png"
    plot_drift_panel(_toy_summary(), out)
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Severity ramp — the shift simulators must scale with severity so the
# monitoring timeline shows genuine onset -> escalation, not a step function.
# --------------------------------------------------------------------------- #

def test_pipeline_break_severity_scales_affected_fraction():
    df = _toy(n=1000)
    rng = np.random.default_rng(1)

    untouched = shift_pipeline_break(df, rng, severity=0.0)
    pd.testing.assert_frame_equal(df, untouched)

    half = shift_pipeline_break(df, rng, severity=0.5)
    frac_default = (half["discharge_disposition_id"] == "1").mean()
    base_frac = (df["discharge_disposition_id"] == "1").mean()
    assert base_frac < frac_default < 1.0

    full = shift_pipeline_break(df, rng, severity=1.0)
    assert set(full["discharge_disposition_id"]) == {"1"}


def test_age_shift_severity_ramps_the_skew():
    df = _toy(n=4000)
    mild = shift_age_older(df, np.random.default_rng(1), severity=0.3)
    strong = shift_age_older(df, np.random.default_rng(1), severity=1.0)
    assert df["age_midpoint"].mean() < mild["age_midpoint"].mean() \
        < strong["age_midpoint"].mean()


def test_prevalence_surge_severity_ramps_positive_rate():
    df = _toy(n=4000)
    mild = shift_prevalence_surge(df, np.random.default_rng(1), severity=0.3)
    strong = shift_prevalence_surge(df, np.random.default_rng(1), severity=1.0)
    assert df["readmitted_lt30"].mean() < mild["readmitted_lt30"].mean() \
        < strong["readmitted_lt30"].mean()


# --------------------------------------------------------------------------- #
# Tiered alerting — WARNING before RETRAIN, and RETRAIN only when sustained.
# --------------------------------------------------------------------------- #

def test_window_status_tiers():
    ref = {"auroc": 0.66, "brier": 0.076}

    ok = window_status({"auroc": 0.66, "brier": 0.077}, ref, drift_share=0.0)
    assert ok["status"] == "ok" and not ok["warnings"]

    # AUROC drop of 0.02 is past 50% of the 0.03 alert -> warning, not retrain.
    warn = window_status({"auroc": 0.64, "brier": 0.077}, ref, drift_share=0.0)
    assert warn["status"] == "warning"
    assert any("AUROC" in w for w in warn["warnings"])

    bad = window_status({"auroc": 0.60, "brier": 0.077}, ref, drift_share=0.0)
    assert bad["status"] == "retrain"
    assert any("AUROC" in r for r in bad["reasons"])


def _win(i, status):
    reasons = ["AUROC fell 0.043 (>=0.03) vs reference"] \
        if status == "retrain" else []
    warnings = ["Brier rise at 0.012 — past 50% of the alert threshold (0.02)"] \
        if status == "warning" else []
    return {"window": i, "status": status, "reasons": reasons,
            "warnings": warnings}


def test_scenario_verdict_requires_sustained_breach():
    # A single breaching window is a WARNING, not a retrain.
    one_off = [_win(1, "ok"), _win(2, "retrain"), _win(3, "ok")]
    v = scenario_verdict(one_off)
    assert v["retrain_recommended"] is False and v["status"] == "warning"

    # Non-consecutive breaches still don't trip the sustained rule.
    scattered = [_win(1, "retrain"), _win(2, "ok"), _win(3, "retrain")]
    assert scenario_verdict(scattered)["retrain_recommended"] is False

    # Two consecutive breaching windows do.
    sustained = [_win(1, "ok"), _win(2, "retrain"), _win(3, "retrain")]
    v = scenario_verdict(sustained)
    assert v["retrain_recommended"] is True and v["status"] == "retrain"
    assert v["consecutive_retrain_windows"] == 2
    assert v["first_flagged_window"] == 2
    assert v["reasons"]

    # All-clear stream stays OK.
    clean = [_win(1, "ok"), _win(2, "ok")]
    v = scenario_verdict(clean)
    assert v["status"] == "ok" and v["reasons"] == []


def test_extract_drifted_columns_handles_both_test_families():
    snapshot = {"metrics": [
        # p-value tests drift when BELOW threshold …
        {"metric_name": "ValueDrift(column=quiet,method=K-S p_value,threshold=0.05)",
         "value": 0.80},
        {"metric_name": "ValueDrift(column=loud,method=chi-square p_value,threshold=0.05)",
         "value": 0.0001},
        # … distance metrics drift when ABOVE it.
        {"metric_name": "ValueDrift(column=moved,method=Jensen-Shannon distance,threshold=0.1)",
         "value": 0.42},
        {"metric_name": "DriftedColumnsCount(drift_share=0.3)",
         "value": {"count": 2.0, "share": 0.66}},
    ]}
    rows = _extract_drifted_columns(snapshot)
    assert [r["column"] for r in rows] == ["moved", "loud", "quiet"]
    assert [r["detected"] for r in rows] == [True, True, False]


def test_plot_drift_timeline_writes_a_figure(tmp_path):
    summary = _toy_summary()
    for name, sc in summary["scenarios"].items():
        breach = sc["retrain_recommended"]
        sc["windows"] = [
            {"window": i, "severity": s,
             "drift_share": sc["drift_share"] * s,
             "auroc": sc["auroc"], "brier": sc["brier"],
             "status": "retrain" if (breach and s >= 1.0) else "ok"}
            for i, s in enumerate([0.0, 0.5, 1.0], start=1)
        ]
    out = tmp_path / "drift_timeline.png"
    plot_drift_timeline(summary, out)
    assert out.exists() and out.stat().st_size > 0
