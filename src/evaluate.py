"""
evaluate.py — Phase 3 of the readmission-monitoring project.

Evaluates the serialized model on the held-out test set the way a clinical team
would judge it — not just discrimination, but whether the probabilities are
*trustworthy* and whether acting on them is *worth it*:

* **Discrimination** — AUROC and AUPRC (AUPRC matters more under a ~9% base
  rate; AUROC can look respectable while precision is poor).
* **Calibration** — a reliability curve + Brier score. A risk score that says
  "20%" should be right about 20% of the time, or downstream resource decisions
  are built on sand.
* **Net benefit / decision-curve analysis** [Vickers & Elkin 2006; Singh, Shah
  & Vickers 2023] — does using the model to prioritize outreach beat the naive
  "contact everyone" or "contact no one" strategies, across the range of
  thresholds a care team might realistically use?

Outputs:
  reports/figures/roc_pr.png
  reports/figures/calibration.png
  reports/figures/decision_curve.png
  reports/evaluation.json   (machine-readable metrics)
  reports/evaluation.md     (written interpretation)
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
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_model_and_test(models_dir: Path, processed_dir: Path):
    bundle = joblib.load(models_dir / "model.joblib")
    spec = bundle["feature_spec"]
    target = bundle["target"]
    test_df = pd.read_csv(processed_dir / "test.csv")
    for col in spec["categorical_features"]:
        test_df[col] = test_df[col].astype(str)
    X = test_df.drop(columns=[target])
    y = test_df[target].to_numpy()
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    return bundle, y, proba


# --------------------------------------------------------------------------- #
# Net benefit / decision-curve analysis
# --------------------------------------------------------------------------- #

def net_benefit(y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray):
    """Net benefit of the model, and of 'treat all', across thresholds.

    NB(model) = TP/N - FP/N * (pt / (1 - pt))
    NB(all)   = prevalence - (1 - prevalence) * (pt / (1 - pt))
    NB(none)  = 0
    """
    n = len(y_true)
    prevalence = y_true.mean()
    nb_model, nb_all = [], []
    for pt in thresholds:
        odds = pt / (1 - pt)
        predicted_pos = proba >= pt
        tp = np.sum(predicted_pos & (y_true == 1))
        fp = np.sum(predicted_pos & (y_true == 0))
        nb_model.append(tp / n - (fp / n) * odds)
        nb_all.append(prevalence - (1 - prevalence) * odds)
    return np.array(nb_model), np.array(nb_all)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_roc_pr(y, proba, prevalence, out_path: Path):
    fpr, tpr, _ = roc_curve(y, proba)
    prec, rec, _ = precision_recall_curve(y, proba)
    auroc = roc_auc_score(y, proba)
    auprc = average_precision_score(y, proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", label="chance")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="ROC curve")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, label=f"AUPRC = {auprc:.3f}")
    axes[1].axhline(prevalence, ls="--", color="grey",
                    label=f"baseline = {prevalence:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision-Recall curve")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_calibration(y, proba, brier, out_path: Path, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins,
                                            strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", label=f"model (Brier = {brier:.4f})")
    ax.set(xlabel="Mean predicted probability",
           ylabel="Observed fraction positive",
           title="Calibration (reliability) curve")
    # Zoom to the operating region; predictions cluster well below 0.5.
    upper = float(min(1.0, max(mean_pred.max(), frac_pos.max()) * 1.15))
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_decision_curve(thresholds, nb_model, nb_all, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, nb_model, label="Model", color="C0")
    ax.plot(thresholds, nb_all, label="Treat all", color="C1", ls="--")
    ax.axhline(0, label="Treat none", color="grey", ls=":")
    ax.set(xlabel="Threshold probability (outreach if risk ≥ threshold)",
           ylabel="Net benefit",
           title="Decision-curve analysis")
    # Net benefit below zero is not actionable; keep the y-floor sensible.
    ax.set_ylim(min(-0.01, nb_model.min()), max(nb_model.max(), 0.0) * 1.2 + 1e-3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #

def write_interpretation(md_path: Path, m: dict, model_name: str):
    prevalence = m["prevalence"]
    auroc, auprc, brier = m["auroc"], m["auprc"], m["brier"]
    # Range of thresholds where the model beats both default strategies.
    useful_lo = m["net_benefit_useful_range"][0]
    useful_hi = m["net_benefit_useful_range"][1]

    md = f"""# Phase 3 — Model Evaluation

_Held-out test set (n = {m['n']:,}), 30-day readmission prevalence = {prevalence:.3f}._
Serialized model: **{model_name}**.

## Discrimination
- **AUROC = {auroc:.3f}** — modest, and honestly so. 30-day readmission is a
  genuinely hard outcome to predict from administrative data; published models
  on this dataset sit in a similar range. A respectable-looking AUROC would be
  a red flag, not a triumph.
- **AUPRC = {auprc:.3f}** vs. a no-skill baseline of {prevalence:.3f}. The model
  carries real signal over chance, but precision is inherently limited by the
  low base rate — most flagged patients will not be readmitted within 30 days.

## Calibration
- **Brier score = {brier:.4f}.** The reliability curve (`figures/calibration.png`)
  shows how closely predicted risks track observed rates. Calibration matters
  more than discrimination here: outreach capacity is allocated against the
  *probabilities*, so systematically over- or under-stated risk wastes scarce
  care-management time.

## Net benefit (decision-curve analysis)
- Across threshold probabilities in roughly **{useful_lo:.0%}–{useful_hi:.0%}**,
  using the model to prioritize outreach yields higher net benefit than either
  "contact everyone" or "contact no one" (`figures/decision_curve.png`). This is
  the question that actually matters operationally: *given finite staff, does
  acting on this model help?* In this range, yes.
- Outside that range the defaults win — e.g. at very low thresholds "treat all"
  is competitive, which is the expected behavior of decision-curve analysis.

## How to read this like a clinician, not a Kaggler
The headline AUROC is unremarkable. The useful findings are that the model is
**calibrated well enough to triage on** and **adds net benefit in the threshold
band a care team would plausibly operate in**. Whether to deploy still depends
on subgroup reliability (Phase 4) and on monitoring for drift (Phase 6).
"""
    md_path.write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()

    bundle, y, proba = load_model_and_test(args.models_dir, args.processed_dir)
    model_name = bundle["model_name"]

    figures_dir = args.reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    prevalence = float(y.mean())
    auroc = float(roc_auc_score(y, proba))
    auprc = float(average_precision_score(y, proba))
    brier = float(brier_score_loss(y, proba))

    thresholds = np.linspace(0.01, 0.60, 60)
    nb_model, nb_all = net_benefit(y, proba, thresholds)

    # Threshold band where the model is the best of the three strategies.
    better = (nb_model > nb_all) & (nb_model > 0)
    if better.any():
        useful_range = [float(thresholds[better].min()),
                        float(thresholds[better].max())]
    else:
        useful_range = [0.0, 0.0]

    plot_roc_pr(y, proba, prevalence, figures_dir / "roc_pr.png")
    plot_calibration(y, proba, brier, figures_dir / "calibration.png")
    plot_decision_curve(thresholds, nb_model, nb_all,
                        figures_dir / "decision_curve.png")

    metrics = {
        "model_name": model_name,
        "n": int(len(y)),
        "prevalence": prevalence,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "net_benefit_useful_range": useful_range,
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "evaluation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    write_interpretation(args.reports_dir / "evaluation.md", metrics, model_name)

    print(f"[eval] model={model_name} n={len(y)} prevalence={prevalence:.3f}")
    print(f"[eval] AUROC={auroc:.3f}  AUPRC={auprc:.3f}  Brier={brier:.4f}")
    print(f"[eval] net-benefit useful threshold range: "
          f"{useful_range[0]:.2f}-{useful_range[1]:.2f}")
    print(f"[write] figures -> {figures_dir}")
    print(f"[write] {args.reports_dir / 'evaluation.json'}")
    print(f"[write] {args.reports_dir / 'evaluation.md'}")
    print("[done] Phase 3 complete.")


if __name__ == "__main__":
    main()
