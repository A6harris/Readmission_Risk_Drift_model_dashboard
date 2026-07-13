"""
drift.py — Phase 6 of the readmission-monitoring project.

This is the heart of the monitoring story. A model that looks fine at validation
degrades silently when the deployment distribution drifts away from training —
the central warning of Finlayson et al. (2021), and the COVID-era sepsis-model
shift documented by Wong et al. (2021). Here we *manufacture* three realistic
shifts and show that they are (a) detectable as data drift and (b) reflected in
model performance decay.

Setup: the held-out test set is split into a **reference** window (the validated
baseline) and a **holdout** pool. Each scenario is then played out as a sequence
of monitoring windows — like weekly batches of production scoring data — in
which the shift ramps in over time:

* **baseline** — resampled holdout, unchanged (a control: drift should be ~0).
* **age_shift** — the population skews progressively older (weighted
  resampling). A demographic shift of the kind that happens when a model meets
  a new catchment area.
* **pipeline_break** — the top-driver field ``discharge_disposition_id``
  collapses to a single default value in a growing fraction of rows, simulating
  a renamed/dropped field rolling out upstream.
* **prevalence_surge** — readmissions become more common (positives
  oversampled), mimicking a COVID-like event where sicker patients return.

Every window is scored against a tiered alert policy — **OK → WARNING →
RETRAIN** — and a retraining recommendation requires the breach to be
*sustained* (>= ``SUSTAINED_WINDOWS`` consecutive windows at retrain severity),
so a single noisy week can't trigger a retrain. For the final, fully-shifted
window we also generate the rich Evidently HTML report and extract *which*
columns drifted, so an alert is attributable, not just a number.

Outputs:
  reports/drift_<scenario>.html      (Evidently visual reports, final window)
  reports/drift_summary.json         (per-scenario timeline + verdict)
  reports/figures/drift_panel.png    (static per-scenario breach panel)
  reports/figures/drift_timeline.png (static metrics-over-time panel)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from evidently import (
    BinaryClassification,
    DataDefinition,
    Dataset,
    Report,
)
from evidently.metrics import DriftedColumnsCount
from evidently.presets import ClassificationQuality, DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

SEED = 42
OUTREACH_THRESHOLD = 0.10  # operating point shared with the fairness audit

# --- Retraining-alert policy. These are policy choices, stated explicitly so
#     the dashboard's OK / WARNING / RETRAIN verdicts are auditable. ---------- #
DRIFT_SHARE_ALERT = 0.30   # >=30% of features drifted
AUROC_DROP_ALERT = 0.03    # AUROC falls >=0.03 below the reference window
BRIER_RISE_ALERT = 0.02    # Brier worsens (rises) >=0.02 above reference
WARN_FRACTION = 0.5        # WARNING when a metric is >=50% of the way to alert
SUSTAINED_WINDOWS = 2      # retrain only after this many consecutive breaches

# --- Monitoring timeline. Each scenario is scored over N_WINDOWS consecutive
#     windows; the shift begins mid-stream and ramps to full severity, so the
#     dashboard shows onset -> escalation -> sustained breach, not a snapshot. - #
WINDOW_SEVERITY = [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]

TOP_DRIFTED_COLUMNS = 8    # how many drifted columns to attribute per scenario


# --------------------------------------------------------------------------- #
# Loading + prediction
# --------------------------------------------------------------------------- #

def load(models_dir: Path, processed_dir: Path):
    bundle = joblib.load(models_dir / "model.joblib")
    spec = bundle["feature_spec"]
    target = bundle["target"]
    test_df = pd.read_csv(processed_dir / "test.csv")
    for col in spec["categorical_features"]:
        test_df[col] = test_df[col].astype(str)
    return bundle, spec, target, test_df


def add_predictions(df: pd.DataFrame, pipeline, target: str) -> pd.DataFrame:
    """Return a copy with a ``pred_proba`` column from the pipeline."""
    X = df.drop(columns=[target])
    out = df.copy()
    out["pred_proba"] = pipeline.predict_proba(X)[:, 1]
    return out


# --------------------------------------------------------------------------- #
# Shift simulators — each takes the holdout frame and a severity in [0, 1]
# (0 = no shift, 1 = the full shift) and returns a shifted copy.
# --------------------------------------------------------------------------- #

def shift_none(df: pd.DataFrame, rng, severity: float = 1.0) -> pd.DataFrame:
    """Control: no change. Drift should be negligible."""
    return df.copy()


def shift_age_older(df: pd.DataFrame, rng, severity: float = 1.0) -> pd.DataFrame:
    """Skew the population older via weighted resampling (whole rows are kept,
    so age band and age_midpoint stay consistent and clinically correlated
    features — diagnoses, medications, disposition — co-shift the way they
    would in a genuinely older catchment area)."""
    if severity <= 0:
        return df.copy()
    # At full severity the age weight is cubed, concentrating mass in the
    # oldest bands — a pronounced, multi-feature demographic shift.
    weights = (df["age_midpoint"].to_numpy(dtype=float) / 50.0) ** (3.0 * severity)
    weights = weights / weights.sum()
    idx = rng.choice(len(df), size=len(df), replace=True, p=weights)
    return df.iloc[idx].reset_index(drop=True)


def shift_pipeline_break(df: pd.DataFrame, rng,
                         severity: float = 1.0) -> pd.DataFrame:
    """Simulate an upstream field collapse: the top-driver discharge field gets
    defaulted to a single value (as if renamed/dropped and back-filled).
    Severity is the fraction of rows affected — a broken upstream job rolling
    out across sites rather than flipping everywhere at once."""
    out = df.copy()
    if severity <= 0:
        return out
    n_broken = int(round(len(out) * min(severity, 1.0)))
    broken_idx = rng.choice(len(out), size=n_broken, replace=False)
    out.iloc[broken_idx, out.columns.get_loc("discharge_disposition_id")] = "1"
    return out


def shift_prevalence_surge(df: pd.DataFrame, rng, severity: float = 1.0,
                           pos_weight: float = 3.0,
                           target: str = "readmitted_lt30") -> pd.DataFrame:
    """Oversample positives so 30-day readmissions become more common. Severity
    ramps the oversampling weight from 1 (no surge) to ``pos_weight``."""
    effective = 1.0 + (pos_weight - 1.0) * max(severity, 0.0)
    w = np.where(df[target].to_numpy() == 1, effective, 1.0)
    w = w / w.sum()
    idx = rng.choice(len(df), size=len(df), replace=True, p=w)
    return df.iloc[idx].reset_index(drop=True)


SCENARIOS = {
    "baseline": shift_none,
    "age_shift": shift_age_older,
    "pipeline_break": shift_pipeline_break,
    "prevalence_surge": shift_prevalence_surge,
}


# --------------------------------------------------------------------------- #
# Evidently reports + metric extraction
# --------------------------------------------------------------------------- #

def build_data_definition(spec: dict, target: str) -> DataDefinition:
    return DataDefinition(
        numerical_columns=spec["numeric_features"],
        categorical_columns=spec["categorical_features"],
        classification=[BinaryClassification(
            target=target, prediction_probas="pred_proba", pos_label=1)],
    )


def _extract_drift_share(snapshot_dict: dict) -> float:
    for metric in snapshot_dict.get("metrics", []):
        if str(metric.get("metric_name", "")).startswith("DriftedColumnsCount"):
            return float(metric["value"]["share"])
    return float("nan")


def _extract_drifted_columns(snapshot_dict: dict) -> list[dict]:
    """Pull per-column drift results out of a DataDriftPreset snapshot so an
    alert can say *which* fields moved.

    Evidently names these metrics like
    ``ValueDrift(column=age,method=K-S p_value,threshold=0.05)``. The stat-test
    differs per column (p-value tests drift when *below* threshold, distance
    metrics when *above*), so detection and the "how far past the line" margin
    are derived per method. Returns [{column, method, score, threshold,
    detected}], drifted columns first, strongest drift first."""
    rows = []
    for metric in snapshot_dict.get("metrics", []):
        name = str(metric.get("metric_name", ""))
        m = re.match(
            r"ValueDrift\(column=([^,)]+)(?:,method=([^,)]+))?"
            r"(?:,threshold=([^,)]+))?\)", name)
        if not m:
            continue
        value = metric.get("value")
        score = float(value) if isinstance(value, (int, float)) else float("nan")
        method = m.group(2) or ""
        threshold = float(m.group(3)) if m.group(3) else float("nan")
        is_p_value = "p_value" in method
        if np.isnan(score) or np.isnan(threshold):
            detected, margin = False, float("-inf")
        elif is_p_value:
            detected, margin = score < threshold, threshold - score
        else:
            detected, margin = score > threshold, score - threshold
        rows.append({"column": m.group(1), "method": method,
                     "score": score, "threshold": threshold,
                     "detected": detected, "_margin": margin})
    rows.sort(key=lambda r: (not r["detected"], -r["_margin"]))
    for r in rows:
        del r["_margin"]
    return rows


def run_drift_check(ref_ds, cur_df, data_definition) -> float:
    """Lightweight per-window drift check: drifted-column share only."""
    cur_ds = Dataset.from_pandas(cur_df, data_definition=data_definition)
    report = Report(metrics=[DriftedColumnsCount(drift_share=DRIFT_SHARE_ALERT)])
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)
    return _extract_drift_share(snapshot.dict())


def run_full_report(ref_ds, cur_df, data_definition,
                    html_path: Path) -> list[dict]:
    """Full Evidently report for the final window: saves the HTML artifact and
    returns the per-column drift attribution. (The alerting drift share comes
    from ``run_drift_check`` so every window is scored identically — the full
    preset also tests the prediction/target columns, which would make the
    final window's share incomparable with the rest of the stream.)"""
    cur_ds = Dataset.from_pandas(cur_df, data_definition=data_definition)
    report = Report(metrics=[
        DataDriftPreset(),
        DriftedColumnsCount(drift_share=DRIFT_SHARE_ALERT),
        ClassificationQuality(),
    ])
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)
    snapshot.save_html(str(html_path))
    return _extract_drifted_columns(snapshot.dict())


def performance(df: pd.DataFrame, target: str) -> dict:
    """Discrimination + calibration on a window, computed with sklearn."""
    y = df[target].to_numpy()
    p = df["pred_proba"].to_numpy()
    has_both = len(np.unique(y)) == 2
    return {
        "n": int(len(df)),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)) if has_both else float("nan"),
        "auprc": float(average_precision_score(y, p)) if has_both else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "alert_rate": float((p >= OUTREACH_THRESHOLD).mean()),
        "mean_pred": float(p.mean()),
    }


# --------------------------------------------------------------------------- #
# Alert policy — per-window tiered status, per-scenario sustained verdict
# --------------------------------------------------------------------------- #

def verdict(scenario_perf: dict, ref_perf: dict, drift_share: float) -> dict:
    """Single-window RETRAIN check: record the reasons that tripped."""
    reasons = []
    if not np.isnan(drift_share) and drift_share >= DRIFT_SHARE_ALERT:
        reasons.append(
            f"data drift: {drift_share:.0%} of features drifted "
            f"(>={DRIFT_SHARE_ALERT:.0%})")
    auroc_drop = ref_perf["auroc"] - scenario_perf["auroc"]
    if not np.isnan(auroc_drop) and auroc_drop >= AUROC_DROP_ALERT:
        reasons.append(
            f"AUROC fell {auroc_drop:.3f} (>={AUROC_DROP_ALERT}) vs reference")
    brier_rise = scenario_perf["brier"] - ref_perf["brier"]
    if brier_rise >= BRIER_RISE_ALERT:
        reasons.append(
            f"Brier rose {brier_rise:.3f} (>={BRIER_RISE_ALERT}) vs reference")
    return {"retrain_recommended": bool(reasons), "reasons": reasons,
            "auroc_drop": float(auroc_drop), "brier_rise": float(brier_rise)}


def window_status(scenario_perf: dict, ref_perf: dict,
                  drift_share: float) -> dict:
    """Tiered status for one monitoring window.

    ``retrain`` when any rule breaches its threshold; ``warning`` when any
    metric is at least ``WARN_FRACTION`` of the way there — early notice that
    something is moving before it becomes a policy breach; ``ok`` otherwise.
    """
    v = verdict(scenario_perf, ref_perf, drift_share)
    warnings = []
    if v["retrain_recommended"]:
        status = "retrain"
    else:
        checks = [
            ("data drift share", drift_share, DRIFT_SHARE_ALERT),
            ("AUROC drop", v["auroc_drop"], AUROC_DROP_ALERT),
            ("Brier rise", v["brier_rise"], BRIER_RISE_ALERT),
        ]
        for label, value, alert_at in checks:
            if not np.isnan(value) and value >= WARN_FRACTION * alert_at:
                warnings.append(
                    f"{label} at {value:.3f} — past {WARN_FRACTION:.0%} of the "
                    f"alert threshold ({alert_at})")
        status = "warning" if warnings else "ok"
    return {"status": status, "warnings": warnings, **v}


def scenario_verdict(windows: list[dict]) -> dict:
    """Roll per-window statuses up to a scenario decision.

    RETRAIN requires >= ``SUSTAINED_WINDOWS`` *consecutive* windows at retrain
    severity — one noisy week never triggers a retrain. A breach that hasn't
    been sustained yet, or any warning-level movement, surfaces as WARNING.
    """
    statuses = [w["status"] for w in windows]
    longest = run = 0
    for s in statuses:
        run = run + 1 if s == "retrain" else 0
        longest = max(longest, run)

    first_flagged = next(
        (w["window"] for w in windows if w["status"] != "ok"), None)
    retrain = longest >= SUSTAINED_WINDOWS
    if retrain:
        status = "retrain"
    elif any(s != "ok" for s in statuses):
        status = "warning"
    else:
        status = "ok"

    # Report reasons from the most recent breaching (or warning) window.
    reasons: list[str] = []
    for w in reversed(windows):
        if w["reasons"]:
            reasons = w["reasons"]
            break
        if not reasons and w["warnings"]:
            reasons = w["warnings"]
    return {
        "status": status,
        "retrain_recommended": retrain,
        "reasons": reasons if retrain else (reasons if status != "ok" else []),
        "first_flagged_window": first_flagged,
        "consecutive_retrain_windows": longest,
    }


# --------------------------------------------------------------------------- #
# Static drift figures — the committable visuals the README/GitHub can show
# --------------------------------------------------------------------------- #

# Scenario display order + short labels, kept stable so the figure reads left to
# right from control -> benign -> harmful.
PLOT_ORDER = ["baseline", "age_shift", "pipeline_break", "prevalence_surge"]
PLOT_LABELS = {
    "baseline": "Baseline\n(control)",
    "age_shift": "Age shift\n(older)",
    "pipeline_break": "Pipeline\nbreak",
    "prevalence_surge": "Prevalence\nsurge",
}
RETRAIN_COLOR = "#c0392b"  # red — alert tripped
WARNING_COLOR = "#e67e22"  # orange — warning tier
HEALTHY_COLOR = "#27ae60"  # green — within policy
TIMELINE_COLORS = {
    "baseline": "#7f8c8d",
    "age_shift": "#2980b9",
    "pipeline_break": "#8e44ad",
    "prevalence_surge": "#c0392b",
}


def plot_drift_panel(summary: dict, out_path: Path) -> None:
    """Render the four scenarios as a three-panel monitoring story.

    Left: data-drift share vs. the alert line. Middle: AUROC vs. the validated
    reference and its drop tolerance. Right: Brier vs. reference and its rise
    tolerance. **Each bar is coloured per panel**: red only in the panel where
    *that* metric breached its threshold, green otherwise. So a red bar always
    means "this is the thing that's broken, right here" — no cross-referencing.
    The breaching bar is annotated with the rule it tripped. This makes the
    central claim ("we distinguish benign from harmful drift, and we can say
    *why*") legible without mentally joining the three panels.
    """
    ref = summary["reference"]
    thr = summary["thresholds"]
    scenarios = summary["scenarios"]
    names = [n for n in PLOT_ORDER if n in scenarios]
    labels = [PLOT_LABELS.get(n, n) for n in names]
    x = np.arange(len(names))

    def bar_color(breached: bool) -> str:
        return RETRAIN_COLOR if breached else HEALTHY_COLOR

    def annotate(ax, xi, top, text):
        """Tag a breaching bar with the rule it tripped, just above the bar."""
        ax.annotate(text, (xi, top), textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=RETRAIN_COLOR)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    # --- Panel 1: data-drift share (breaches when >= drift_share_alert) ---- #
    drift_vals = [scenarios[n]["drift_share"] * 100 for n in names]
    drift_breach = [scenarios[n]["drift_share"] >= thr["drift_share_alert"]
                    for n in names]
    axes[0].bar(x, drift_vals, color=[bar_color(b) for b in drift_breach])
    axes[0].axhline(thr["drift_share_alert"] * 100, ls=":", color="grey",
                    label=f"alert ≥ {thr['drift_share_alert']:.0%}")
    for xi, v, b in zip(x, drift_vals, drift_breach):
        if b:
            annotate(axes[0], xi, v, "≥ alert")
    axes[0].set_ylim(0, max(drift_vals + [thr["drift_share_alert"] * 100]) * 1.25)
    axes[0].set(ylabel="% of features drifted", title="Data drift")
    axes[0].legend(loc="upper right", fontsize=9)

    # --- Panel 2: AUROC, higher is better (breaches when drop >= alert) ---- #
    auroc_vals = [scenarios[n]["auroc"] for n in names]
    auroc_drop = [ref["auroc"] - scenarios[n]["auroc"] for n in names]
    auroc_breach = [d >= thr["auroc_drop_alert"] for d in auroc_drop]
    axes[1].bar(x, auroc_vals, color=[bar_color(b) for b in auroc_breach])
    axes[1].axhline(ref["auroc"], ls="--", color="grey",
                    label=f"reference = {ref['auroc']:.3f}")
    axes[1].axhline(ref["auroc"] - thr["auroc_drop_alert"], ls=":", color="grey",
                    label=f"alert ≤ {ref['auroc'] - thr['auroc_drop_alert']:.3f}")
    for xi, v, d, b in zip(x, auroc_vals, auroc_drop, auroc_breach):
        if b:
            annotate(axes[1], xi, v, f"−{d:.3f}")
    # Zoom to the band where the bars and threshold lines actually live.
    lo = min(auroc_vals + [ref["auroc"] - thr["auroc_drop_alert"]]) - 0.02
    axes[1].set_ylim(max(0.0, lo), max(auroc_vals + [ref["auroc"]]) + 0.015)
    axes[1].set(ylabel="AUROC", title="Discrimination")
    axes[1].legend(loc="lower left", fontsize=9)

    # --- Panel 3: Brier, lower is better (breaches when rise >= alert) ----- #
    brier_vals = [scenarios[n]["brier"] for n in names]
    brier_rise = [scenarios[n]["brier"] - ref["brier"] for n in names]
    brier_breach = [r >= thr["brier_rise_alert"] for r in brier_rise]
    axes[2].bar(x, brier_vals, color=[bar_color(b) for b in brier_breach])
    axes[2].axhline(ref["brier"], ls="--", color="grey",
                    label=f"reference = {ref['brier']:.4f}")
    axes[2].axhline(ref["brier"] + thr["brier_rise_alert"], ls=":", color="grey",
                    label=f"alert ≥ {ref['brier'] + thr['brier_rise_alert']:.4f}")
    for xi, v, r, b in zip(x, brier_vals, brier_rise, brier_breach):
        if b:
            annotate(axes[2], xi, v, f"+{r:.3f}")
    axes[2].set_ylim(0, max(brier_vals) * 1.18)
    axes[2].set(ylabel="Brier (lower = better)", title="Calibration")
    axes[2].legend(loc="upper left", fontsize=9)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)

    # Shared legend: red now means "this metric breached its threshold".
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=HEALTHY_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=RETRAIN_COLOR),
    ]
    fig.suptitle("Drift monitoring: each panel flags the scenarios that breach "
                 "its rule", fontsize=13, y=1.10)
    fig.legend(legend_handles,
               ["within policy", "breached threshold → RETRAIN"],
               loc="center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9,
               frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_drift_timeline(summary: dict, out_path: Path) -> None:
    """Render the monitoring timeline: drift share, AUROC, and Brier per
    window, one line per scenario, with the alert thresholds drawn in. This is
    the "what monitoring actually looks like" figure — onset, escalation, and
    the level at which policy trips — rather than a single before/after bar.
    """
    ref = summary["reference"]
    thr = summary["thresholds"]
    scenarios = summary["scenarios"]
    names = [n for n in PLOT_ORDER
             if n in scenarios and scenarios[n].get("windows")]
    if not names:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
    # (metric key, ylabel, title, alert line, label, this-panel breach test)
    panels = [
        ("drift_share", "Share of features drifted", "Data drift",
         thr["drift_share_alert"], "alert ≥ {:.0%}",
         lambda w: w["drift_share"] >= thr["drift_share_alert"]),
        ("auroc", "AUROC", "Discrimination",
         ref["auroc"] - thr["auroc_drop_alert"], "alert ≤ {:.3f}",
         lambda w: ref["auroc"] - w["auroc"] >= thr["auroc_drop_alert"]),
        ("brier", "Brier (lower = better)", "Calibration",
         ref["brier"] + thr["brier_rise_alert"], "alert ≥ {:.4f}",
         lambda w: w["brier"] - ref["brier"] >= thr["brier_rise_alert"]),
    ]
    ref_lines = {"auroc": ref["auroc"], "brier": ref["brier"], "drift_share": None}

    for ax, (key, ylabel, title, alert_level, alert_fmt, breached) in zip(
            axes, panels):
        for name in names:
            windows = scenarios[name]["windows"]
            xs = [w["window"] for w in windows]
            ys = [w[key] for w in windows]
            color = TIMELINE_COLORS.get(name, "#333333")
            ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=color,
                    label=PLOT_LABELS.get(name, name).replace("\n", " "))
            # Ring only the windows where *this panel's* rule breached, so a
            # red ring always means "this is what's broken, right here".
            breach_x = [w["window"] for w in windows if breached(w)]
            breach_y = [w[key] for w in windows if breached(w)]
            if breach_x:
                ax.scatter(breach_x, breach_y, s=110, facecolors="none",
                           edgecolors=RETRAIN_COLOR, linewidths=1.6, zorder=5)
        if ref_lines[key] is not None:
            ax.axhline(ref_lines[key], ls="--", color="grey", lw=1,
                       label="reference")
        ax.axhline(alert_level, ls=":", color=RETRAIN_COLOR, lw=1.2,
                   label=alert_fmt.format(alert_level))
        ax.set(title=title, ylabel=ylabel, xlabel="monitoring window")
        ax.set_xticks([w["window"] for w in scenarios[names[0]]["windows"]])

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.suptitle("Monitoring timeline: drift ramps in mid-stream; red rings mark "
                 "windows that breach that panel's rule", fontsize=13, y=1.12)
    fig.legend(handles, labels_, loc="center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(labels_), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()

    bundle, spec, target, test_df = load(args.models_dir, args.processed_dir)
    pipeline = bundle["pipeline"]
    data_definition = build_data_definition(spec, target)
    rng = np.random.default_rng(SEED)

    # Split the test set into reference (validated baseline) and holdout pool.
    shuffled = test_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    half = len(shuffled) // 2
    reference = add_predictions(shuffled.iloc[:half], pipeline, target)
    holdout = shuffled.iloc[half:].reset_index(drop=True)
    ref_ds = Dataset.from_pandas(reference, data_definition=data_definition)

    ref_perf = performance(reference, target)
    print(f"[ref] n={ref_perf['n']} AUROC={ref_perf['auroc']:.3f} "
          f"Brier={ref_perf['brier']:.4f} prevalence={ref_perf['prevalence']:.3f}")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "reference": ref_perf,
        "thresholds": {
            "drift_share_alert": DRIFT_SHARE_ALERT,
            "auroc_drop_alert": AUROC_DROP_ALERT,
            "brier_rise_alert": BRIER_RISE_ALERT,
            "outreach_threshold": OUTREACH_THRESHOLD,
            "warn_fraction": WARN_FRACTION,
            "sustained_windows": SUSTAINED_WINDOWS,
        },
        "scenarios": {},
    }

    n_windows = len(WINDOW_SEVERITY)
    for name, shifter in SCENARIOS.items():
        windows = []
        final_cur = None
        for i, severity in enumerate(WINDOW_SEVERITY, start=1):
            is_final = i == n_windows
            # Earlier windows bootstrap the holdout so healthy weeks show
            # realistic sampling wiggle; the final window is the full holdout
            # under the full shift (the cleanest before/after comparison).
            base = holdout if is_final else holdout.sample(
                frac=1.0, replace=True,
                random_state=int(rng.integers(0, 2**31)),
            ).reset_index(drop=True)
            cur = add_predictions(shifter(base, rng, severity=severity),
                                  pipeline, target)
            drift_share = run_drift_check(ref_ds, cur, data_definition)
            perf = performance(cur, target)
            ws = window_status(perf, ref_perf, drift_share)
            windows.append({"window": i, "severity": severity,
                            "drift_share": drift_share, **perf, **ws})
            if is_final:
                final_cur = cur

        # Rich Evidently report + per-column attribution for the final window.
        html_path = args.reports_dir / f"drift_{name}.html"
        drifted_cols = run_full_report(
            ref_ds, final_cur, data_definition, html_path)
        final = windows[-1]

        sv = scenario_verdict(windows)
        summary["scenarios"][name] = {
            "html": html_path.name,
            "drift_share": final["drift_share"],
            **{k: final[k] for k in ("n", "prevalence", "auroc", "auprc",
                                     "brier", "alert_rate", "mean_pred",
                                     "auroc_drop", "brier_rise")},
            **sv,
            "windows": windows,
            "top_drifted_columns": drifted_cols[:TOP_DRIFTED_COLUMNS],
        }
        trail = "".join({"ok": ".", "warning": "w", "retrain": "R"}[w["status"]]
                        for w in windows)
        print(f"[{name:16s}] windows={trail} drift={final['drift_share']:.2f} "
              f"AUROC={final['auroc']:.3f} Brier={final['brier']:.4f} "
              f"-> {sv['status'].upper()}")

    (args.reports_dir / "drift_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[write] {args.reports_dir / 'drift_summary.json'}")
    print(f"[write] Evidently HTML reports -> {args.reports_dir}")

    figures_dir = args.reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_drift_panel(summary, figures_dir / "drift_panel.png")
    plot_drift_timeline(summary, figures_dir / "drift_timeline.png")
    print(f"[write] figures -> {figures_dir / 'drift_panel.png'}, "
          f"{figures_dir / 'drift_timeline.png'}")
    print("[done] Phase 6 complete.")


if __name__ == "__main__":
    main()
