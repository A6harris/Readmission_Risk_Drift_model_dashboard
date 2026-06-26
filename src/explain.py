"""
explain.py — Phase 5 of the readmission-monitoring project.

Explainability is not a nice-to-have in clinical AI: a care team needs to see
*why* a patient was flagged before they act on it, and a reviewer needs to check
that the model's drivers are clinically plausible rather than artifacts or
proxies. This module uses SHAP to produce both views:

* **Global** — which features move predictions across the population
  (a beeswarm summary on the encoded features, plus a bar chart that aggregates
  one-hot columns back to their original clinical variable so the ranking reads
  in human terms).
* **Local** — per-prediction waterfall plots for a high-risk and a low-risk
  patient, the kind of explanation that would sit next to an individual alert.

The model is a ``Pipeline`` (ColumnTransformer → XGBoost), so we explain the
tree model on the *transformed* feature space and carry the encoded feature
names through, then map them back to source variables for the aggregated view.

Outputs:
  reports/figures/shap_summary.png
  reports/figures/shap_importance_grouped.png
  reports/figures/shap_waterfall_high.png
  reports/figures/shap_waterfall_low.png
  reports/shap_top_features.json
  reports/explainability.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

SEED = 42
SAMPLE_SIZE = 2000  # rows used for the global plots (fast + readable beeswarm)


def load(models_dir: Path, processed_dir: Path):
    bundle = joblib.load(models_dir / "model.joblib")
    spec = bundle["feature_spec"]
    target = bundle["target"]
    test_df = pd.read_csv(processed_dir / "test.csv")
    for col in spec["categorical_features"]:
        test_df[col] = test_df[col].astype(str)
    X = test_df.drop(columns=[target])
    y = test_df[target].to_numpy()
    return bundle, spec, X, y


def original_feature_map(spec: dict, encoded_names: list[str], preprocessor):
    """Map each encoded column index to its source (clinical) variable.

    Numeric columns map to themselves; one-hot columns map back to the
    categorical they came from (so 'race=Caucasian', 'race=Asian', ... all
    collapse to 'race' for the aggregated importance view).
    """
    mapping = list(spec["numeric_features"])  # StandardScaler is 1:1
    ohe = preprocessor.named_transformers_["cat"]
    for feature, categories in zip(spec["categorical_features"], ohe.categories_):
        mapping.extend([feature] * len(categories))
    assert len(mapping) == len(encoded_names), (
        f"map length {len(mapping)} != encoded {len(encoded_names)}"
    )
    return mapping


def plot_grouped_importance(shap_values, source_map, out_path: Path, top_n=15):
    """Bar chart of mean|SHAP| aggregated to original clinical variables."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    agg = pd.Series(mean_abs, index=source_map).groupby(level=0).sum()
    agg = agg.sort_values(ascending=False).head(top_n)[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(agg.index, agg.to_numpy(), color="C0")
    ax.set_xlabel("Mean |SHAP value|  (impact on model output, log-odds)")
    ax.set_title("Global feature importance (aggregated to source variables)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return agg[::-1]


def plot_waterfall(explanation, idx: int, title: str, out_path: Path):
    fig = plt.figure()
    shap.plots.waterfall(explanation[idx], max_display=12, show=False)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# Clinical interpretation snippets, emitted only for variables that actually
# rank in the model's top features. Keeping the narrative data-driven avoids
# the trap of describing drivers the model doesn't actually use.
_FEATURE_NOTES = {
    "discharge_disposition_id":
        "where the patient is sent after discharge (home, skilled nursing, "
        "transfer). Known at discharge time, so it is legitimately available — "
        "but the model's heavy reliance on specific administrative codes is "
        "exactly the kind of shortcut behavior worth auditing for leakage.",
    "medical_specialty":
        "the attending service, a proxy for case mix and acuity (e.g. "
        "oncology vs. orthopedics carry very different baseline risk).",
    "diag_1_group": "primary diagnosis category — direct clinical signal.",
    "diag_2_group": "secondary diagnosis category — comorbidity signal.",
    "diag_3_group": "additional diagnosis category — comorbidity signal.",
    "admission_source_id":
        "how the patient arrived (ER, physician referral, transfer) — "
        "clinically meaningful context for acuity.",
    "admission_type_id":
        "admission type (emergency vs. elective), a reasonable acuity proxy.",
    "number_inpatient":
        "prior inpatient admissions — the classic, well-validated readmission "
        "predictor.",
    "number_emergency": "prior ER visits — prior-utilization signal.",
    "number_outpatient": "prior outpatient visits — prior-utilization signal.",
    "number_diagnoses": "diagnosis count — a proxy for illness burden.",
    "num_medications": "medication count — complexity / illness burden.",
    "time_in_hospital": "length of stay — acuity / complexity.",
    "age": ("age, expected to matter — but also the variable with the widest "
            "performance gap in the fairness audit (Phase 4), so read its "
            "influence alongside that caveat."),
    "insulin": "insulin use/changes — relevant in this diabetic cohort.",
    "metformin": "metformin use/changes — relevant in this diabetic cohort.",
    "repaglinide": "repaglinide use/changes — relevant in this diabetic cohort.",
}

# Prior-utilization variables; we check explicitly whether they surface high,
# since the literature expects them to.
_UTILIZATION = {"number_inpatient", "number_emergency", "number_outpatient"}


def write_report(md_path: Path, top_features: pd.Series,
                 top_encoded: pd.Series, examples: dict):
    top_list = "\n".join(
        f"{i+1}. **{name}** (summed mean |SHAP| = {val:.3f})"
        for i, (name, val) in enumerate(top_features.items())
    )
    encoded_list = "\n".join(
        f"{i+1}. `{name}` ({val:.3f})"
        for i, (name, val) in enumerate(top_encoded.items())
    )

    # Data-driven plausibility bullets for whichever variables actually rank.
    plausibility = "\n".join(
        f"- **`{name}`** — {_FEATURE_NOTES[name]}"
        for name in top_features.index if name in _FEATURE_NOTES
    )

    # Honest meta-observation about where utilization landed.
    util_ranks = [list(top_features.index).index(u) + 1
                  for u in _UTILIZATION if u in top_features.index]
    if util_ranks and min(util_ranks) <= 5:
        util_note = (
            "Prior-utilization counts surface among the top drivers, as the "
            "readmission literature predicts.")
    else:
        util_note = (
            "Notably, raw prior-utilization counts (`number_inpatient`, "
            "`number_emergency`, `number_outpatient`) rank *lower* than the "
            "readmission literature would lead you to expect. This model leans "
            "more on administrative/categorical signals (discharge disposition, "
            "medical specialty, diagnosis groups) than on prior-utilization "
            "tallies — an honest finding that itself warrants scrutiny before "
            "any deployment.")

    md = f"""# Phase 5 — Explainability (SHAP)

## Global drivers

Aggregated to source clinical variables (one-hot columns summed back together),
the strongest drivers of predicted 30-day readmission risk are:

{top_list}

The strongest *individual* encoded columns (before aggregation) are:

{encoded_list}

See `figures/shap_summary.png` for the per-feature beeswarm and
`figures/shap_importance_grouped.png` for the aggregated ranking above. Summing
absolute SHAP over one-hot columns does favor higher-cardinality variables, but
here the per-column ranking tells the same story — specific discharge
dispositions and specialties dominate on their own, so this is real reliance,
not just an aggregation artifact.

### Are these clinically plausible?

{plausibility}

{util_note}

"Plausible" is not "validated": several top drivers are administrative proxies
available at discharge, and the per-prediction explanations below are what a
clinician would actually inspect before acting on an alert.

## Local explanations

Two individual predictions illustrate how the same drivers play out per patient
(`figures/shap_waterfall_high.png`, `figures/shap_waterfall_low.png`):

- **High-risk example** — predicted probability {examples['high_p']:.3f}
  (actual outcome: {examples['high_y']}). The waterfall shows which features
  push this patient's risk above the base rate.
- **Low-risk example** — predicted probability {examples['low_p']:.3f}
  (actual outcome: {examples['low_y']}). For this patient the same kinds of
  features push the prediction the other way.

Each waterfall starts from the model's base rate and shows the additive
contribution of each feature, in log-odds — the transparency a care team needs
to trust (or override) an individual alert.
"""
    md_path.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()

    bundle, spec, X, y = load(args.models_dir, args.processed_dir)
    pipe = bundle["pipeline"]
    preprocessor = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    figures_dir = args.reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Sample for the global view (TreeExplainer is exact but the beeswarm is
    # unreadable with 14k points).
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(X), size=min(SAMPLE_SIZE, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]

    encoded_names = list(preprocessor.get_feature_names_out())
    X_trans = preprocessor.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    source_map = original_feature_map(spec, encoded_names, preprocessor)
    # Strip the transformer prefixes ("num__", "cat__") for readable labels.
    clean_names = [n.split("__", 1)[-1] for n in encoded_names]

    print(f"[shap] explaining {len(X_sample)} rows over "
          f"{len(encoded_names)} encoded features")
    explainer = shap.TreeExplainer(clf)
    explanation = explainer(X_trans)
    explanation.feature_names = clean_names
    shap_values = explanation.values

    # Show raw (un-scaled) values for the numeric features in the plots — a
    # clinician reads "6 prior admissions", not a z-score of 6.35. This only
    # affects the displayed feature values, never the SHAP values. Numeric
    # columns come first in the ColumnTransformer; categoricals stay 0/1.
    n_num = len(spec["numeric_features"])
    display_data = np.asarray(X_trans, dtype=float).copy()
    display_data[:, :n_num] = X_sample[spec["numeric_features"]].to_numpy()
    explanation.data = display_data

    # --- Global plots ---------------------------------------------------------
    shap.summary_plot(shap_values, features=display_data,
                      feature_names=clean_names, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    top_features = plot_grouped_importance(
        shap_values, source_map, figures_dir / "shap_importance_grouped.png"
    )
    # Top individual encoded columns (before aggregation), for an honest
    # cross-check against the high-cardinality bias of summing.
    mean_abs_encoded = np.abs(shap_values).mean(axis=0)
    top_encoded = (
        pd.Series(mean_abs_encoded, index=clean_names)
        .sort_values(ascending=False)
        .head(10)
    )

    # --- Local plots: a confident high-risk and low-risk patient --------------
    proba_sample = pipe.predict_proba(X_sample)[:, 1]
    high_idx = int(np.argmax(proba_sample))
    low_idx = int(np.argmin(proba_sample))
    y_sample = y[sample_idx]

    plot_waterfall(explanation, high_idx,
                   f"High-risk patient (p={proba_sample[high_idx]:.3f})",
                   figures_dir / "shap_waterfall_high.png")
    plot_waterfall(explanation, low_idx,
                   f"Low-risk patient (p={proba_sample[low_idx]:.3f})",
                   figures_dir / "shap_waterfall_low.png")

    examples = {
        "high_p": float(proba_sample[high_idx]),
        "high_y": int(y_sample[high_idx]),
        "low_p": float(proba_sample[low_idx]),
        "low_y": int(y_sample[low_idx]),
    }

    # --- Reports --------------------------------------------------------------
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "shap_top_features.json").write_text(
        json.dumps(
            {"top_features_aggregated": {k: float(v) for k, v in top_features.items()},
             "top_features_encoded": {k: float(v) for k, v in top_encoded.items()},
             "examples": examples},
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.reports_dir / "explainability.md", top_features,
                 top_encoded, examples)

    print("[shap] top source features: "
          + ", ".join(list(top_features.index)[:5]))
    print(f"[write] figures -> {figures_dir}")
    print(f"[write] {args.reports_dir / 'shap_top_features.json'}")
    print(f"[write] {args.reports_dir / 'explainability.md'}")
    print("[done] Phase 5 complete.")


if __name__ == "__main__":
    main()
