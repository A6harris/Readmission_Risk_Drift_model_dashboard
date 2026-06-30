"""
drift.py — Phase 6 of the readmission-monitoring project.

This is the heart of the monitoring story. A model that looks fine at validation
degrades silently when the deployment distribution drifts away from training —
the central warning of Finlayson et al. (2021), and the COVID-era sepsis-model
shift documented by Wong et al. (2021). Here we *manufacture* three realistic
shifts and show that they are (a) detectable as data drift and (b) reflected in
model performance decay.

Setup: the held-out test set is split into a **reference** window (the validated
baseline) and a **holdout** window. We then derive several "current" windows
from the holdout:

* **baseline** — the holdout unchanged (a control: drift should be ~0).
* **age_shift** — the population skews older (weighted resampling). A demographic
  shift of the kind that happens when a model meets a new catchment area.
* **pipeline_break** — the top-driver field ``discharge_disposition_id`` collapses
  to a single default value, simulating a renamed/dropped field upstream.
* **prevalence_surge** — readmissions become more common (positives oversampled),
  mimicking a COVID-like event where sicker patients return.

For each scenario we generate an Evidently data-drift + classification report
(the rich HTML artifact) and independently compute performance metrics with
scikit-learn. A scenario trips **RETRAIN RECOMMENDED** when drift is widespread
or performance decays past set thresholds — the signal the dashboard surfaces.

Outputs:
  reports/drift_<scenario>.html   (Evidently visual reports)
  reports/drift_summary.json      (per-scenario drift + performance + verdict)
  reports/figures/drift_panel.png (the static, committable drift visual)
"""

from __future__ import annotations

import argparse
import json
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

# --- Retraining-alert thresholds. These are policy choices, stated explicitly
#     so the dashboard's "RETRAIN RECOMMENDED" verdict is auditable. ---------- #
DRIFT_SHARE_ALERT = 0.30   # >=30% of features drifted
AUROC_DROP_ALERT = 0.03    # AUROC falls >=0.03 below the reference window
BRIER_RISE_ALERT = 0.02    # Brier worsens (rises) >=0.02 above reference


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
# Shift simulators — each takes the holdout frame and returns a shifted copy
# --------------------------------------------------------------------------- #

def shift_none(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Control: no change. Drift should be negligible."""
    return df.copy()


def shift_age_older(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Skew the population strongly older via weighted resampling (whole rows
    are kept, so age band and age_midpoint stay consistent and clinically
    correlated features — diagnoses, medications, disposition — co-shift the way
    they would in a genuinely older catchment area)."""
    # Cube the age weight so mass concentrates in the oldest bands, producing a
    # pronounced, multi-feature demographic shift rather than a token nudge.
    weights = (df["age_midpoint"].to_numpy(dtype=float) / 50.0) ** 3
    weights = weights / weights.sum()
    idx = rng.choice(len(df), size=len(df), replace=True, p=weights)
    return df.iloc[idx].reset_index(drop=True)


def shift_pipeline_break(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Simulate an upstream field collapse: the top-driver discharge field gets
    defaulted to a single value (as if renamed/dropped and back-filled)."""
    out = df.copy()
    out["discharge_disposition_id"] = "1"
    return out


def shift_prevalence_surge(df: pd.DataFrame, rng, pos_weight: float = 3.0,
                           target: str = "readmitted_lt30") -> pd.DataFrame:
    """Oversample positives so 30-day readmissions become more common."""
    w = np.where(df[target].to_numpy() == 1, pos_weight, 1.0)
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
# Evidently report + metric extraction
# --------------------------------------------------------------------------- #

def build_data_definition(spec: dict, target: str) -> DataDefinition:
    return DataDefinition(
        numerical_columns=spec["numeric_features"],
        categorical_columns=spec["categorical_features"],
        classification=[BinaryClassification(
            target=target, prediction_probas="pred_proba", pos_label=1)],
    )


def run_evidently(ref_df, cur_df, data_definition, html_path: Path) -> float:
    """Run the Evidently report, save HTML, return the drifted-column share."""
    ref_ds = Dataset.from_pandas(ref_df, data_definition=data_definition)
    cur_ds = Dataset.from_pandas(cur_df, data_definition=data_definition)
    report = Report(metrics=[
        DataDriftPreset(),
        DriftedColumnsCount(drift_share=DRIFT_SHARE_ALERT),
        ClassificationQuality(),
    ])
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)
    snapshot.save_html(str(html_path))

    drift_share = float("nan")
    for metric in snapshot.dict().get("metrics", []):
        if str(metric.get("metric_name", "")).startswith("DriftedColumnsCount"):
            drift_share = float(metric["value"]["share"])
            break
    return drift_share


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


def verdict(scenario_perf: dict, ref_perf: dict, drift_share: float) -> dict:
    """Decide RETRAIN RECOMMENDED and record the reasons that tripped."""
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


# --------------------------------------------------------------------------- #
# Static drift figure — the committable visual the README/GitHub can show
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
HEALTHY_COLOR = "#27ae60"  # green — within policy


def plot_drift_panel(summary: dict, out_path: Path) -> None:
    """Render the four scenarios as a three-panel monitoring story.

    Left: data-drift share vs. the alert line. Middle: AUROC vs. the validated
    reference and its drop tolerance. Right: Brier vs. reference and its rise
    tolerance. Bars are coloured by the retrain verdict, so the eye lands first
    on the scenarios that trip ``RETRAIN RECOMMENDED`` — making the central
    claim ("we distinguish benign from harmful drift") legible at a glance.
    """
    ref = summary["reference"]
    thr = summary["thresholds"]
    scenarios = summary["scenarios"]
    names = [n for n in PLOT_ORDER if n in scenarios]
    labels = [PLOT_LABELS.get(n, n) for n in names]
    colors = [RETRAIN_COLOR if scenarios[n]["retrain_recommended"]
              else HEALTHY_COLOR for n in names]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    # --- Panel 1: data-drift share ---------------------------------------- #
    drift_vals = [scenarios[n]["drift_share"] * 100 for n in names]
    axes[0].bar(x, drift_vals, color=colors)
    axes[0].axhline(thr["drift_share_alert"] * 100, ls="--", color="grey",
                    label=f"alert ≥ {thr['drift_share_alert']:.0%}")
    axes[0].set(ylabel="% of features drifted", title="Data drift")
    axes[0].legend(loc="upper left", fontsize=9)

    # --- Panel 2: AUROC (higher is better) -------------------------------- #
    auroc_vals = [scenarios[n]["auroc"] for n in names]
    axes[1].bar(x, auroc_vals, color=colors)
    axes[1].axhline(ref["auroc"], ls="--", color="grey",
                    label=f"reference = {ref['auroc']:.3f}")
    axes[1].axhline(ref["auroc"] - thr["auroc_drop_alert"], ls=":", color="grey",
                    label=f"alert ≤ {ref['auroc'] - thr['auroc_drop_alert']:.3f}")
    # Zoom to the band where the bars and threshold lines actually live.
    lo = min(auroc_vals + [ref["auroc"] - thr["auroc_drop_alert"]]) - 0.02
    axes[1].set_ylim(max(0.0, lo), max(auroc_vals + [ref["auroc"]]) + 0.01)
    axes[1].set(ylabel="AUROC", title="Discrimination")
    axes[1].legend(loc="lower left", fontsize=9)

    # --- Panel 3: Brier (lower is better) --------------------------------- #
    brier_vals = [scenarios[n]["brier"] for n in names]
    axes[2].bar(x, brier_vals, color=colors)
    axes[2].axhline(ref["brier"], ls="--", color="grey",
                    label=f"reference = {ref['brier']:.4f}")
    axes[2].axhline(ref["brier"] + thr["brier_rise_alert"], ls=":", color="grey",
                    label=f"alert ≥ {ref['brier'] + thr['brier_rise_alert']:.4f}")
    axes[2].set(ylabel="Brier (lower = better)", title="Calibration")
    axes[2].legend(loc="upper left", fontsize=9)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)

    # A shared legend for the verdict colour-coding.
    verdict_handles = [
        plt.Rectangle((0, 0), 1, 1, color=HEALTHY_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=RETRAIN_COLOR),
    ]
    fig.legend(verdict_handles, ["healthy", "RETRAIN recommended"],
               loc="upper right", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Drift monitoring: which simulated shifts trip a retraining alert",
                 fontsize=13, y=1.02)
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

    # Split the test set into reference (validated baseline) and holdout windows.
    shuffled = test_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    half = len(shuffled) // 2
    reference = add_predictions(shuffled.iloc[:half], pipeline, target)
    holdout = shuffled.iloc[half:].reset_index(drop=True)

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
        },
        "scenarios": {},
    }

    for name, shifter in SCENARIOS.items():
        cur_raw = shifter(holdout, rng)
        cur = add_predictions(cur_raw, pipeline, target)
        html_path = args.reports_dir / f"drift_{name}.html"
        drift_share = run_evidently(reference, cur, data_definition, html_path)
        perf = performance(cur, target)
        v = verdict(perf, ref_perf, drift_share)
        summary["scenarios"][name] = {
            "html": html_path.name,
            "drift_share": drift_share,
            **perf,
            **v,
        }
        flag = "RETRAIN" if v["retrain_recommended"] else "ok"
        print(f"[{name:16s}] drift={drift_share:.2f} AUROC={perf['auroc']:.3f} "
              f"Brier={perf['brier']:.4f} prev={perf['prevalence']:.3f} -> {flag}")

    (args.reports_dir / "drift_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[write] {args.reports_dir / 'drift_summary.json'}")
    print(f"[write] Evidently HTML reports -> {args.reports_dir}")

    figures_dir = args.reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_drift_panel(summary, figures_dir / "drift_panel.png")
    print(f"[write] figure -> {figures_dir / 'drift_panel.png'}")
    print("[done] Phase 6 complete.")


if __name__ == "__main__":
    main()
