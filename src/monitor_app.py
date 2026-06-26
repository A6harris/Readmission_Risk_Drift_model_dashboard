"""
monitor_app.py — Phase 7 of the readmission-monitoring project.

The Streamlit monitoring dashboard. It reads the artifacts produced by the
earlier phases and presents them as three tabs:

* **Model overview** — discrimination, calibration, net benefit, SHAP drivers.
* **Fairness** — per-subgroup reliability across race / gender / age.
* **Monitoring** — pick a drift scenario and watch performance decay; a clear
  **RETRAIN RECOMMENDED** banner trips when thresholds are breached.

The app is a *reader* of artifacts, not a trainer: run the pipeline first
(``data_prep`` → ``train`` → ``evaluate`` → ``fairness`` → ``explain`` →
``drift``) and this dashboard visualizes the results. It degrades gracefully,
telling you which step to run if an artifact is missing.

    streamlit run src/monitor_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

st.set_page_config(
    page_title="Readmission Risk — Monitoring",
    page_icon="🏥",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Artifact loading (cached) with graceful fallbacks
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def need(artifact, run_hint: str) -> bool:
    """Render a 'run this step' message and return True if the artifact is None."""
    if artifact is None:
        st.warning(f"Missing artifact — run `{run_hint}` to generate it.")
        return True
    return False


def show_figure(path: Path, caption: str | None = None):
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure not found: `{path.name}` — run the matching phase.")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("🏥 Readmission Risk — Model & Drift Monitoring")
st.caption(
    "A 30-day hospital readmission model wrapped in a responsible-deployment "
    "layer: fairness auditing, explainability, calibration & net-benefit "
    "analysis, and continuous drift monitoring. **The point isn't the AUC — "
    "it's everything around it.**"
)

metrics = load_json(MODELS_DIR / "metrics.json")
evaluation = load_json(REPORTS_DIR / "evaluation.json")
fairness = load_json(REPORTS_DIR / "fairness.json")
shap_top = load_json(REPORTS_DIR / "shap_top_features.json")
drift = load_json(REPORTS_DIR / "drift_summary.json")

tab_overview, tab_fairness, tab_monitor = st.tabs(
    ["📊 Model overview", "⚖️ Fairness", "🚨 Monitoring"]
)


# --------------------------------------------------------------------------- #
# Tab 1 — Model overview
# --------------------------------------------------------------------------- #

with tab_overview:
    st.subheader("How well does the model discriminate — and is it trustworthy?")
    if not need(evaluation, "python src/evaluate.py"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUROC", f"{evaluation['auroc']:.3f}")
        c2.metric("AUPRC", f"{evaluation['auprc']:.3f}",
                  help=f"No-skill baseline = prevalence = {evaluation['prevalence']:.3f}")
        c3.metric("Brier score", f"{evaluation['brier']:.4f}",
                  help="Lower is better; calibration quality")
        c4.metric("Model", str(evaluation.get("model_name", "—")))

        st.info(
            "AUROC is modest **on purpose** — 30-day readmission is genuinely "
            "hard to predict, and a suspiciously high number would be the red "
            "flag. What matters more: the model is **calibrated** (Brier beats "
            "the no-skill baseline) and adds **net benefit** across the "
            "actionable threshold band.",
            icon="🩺",
        )

        col_left, col_right = st.columns(2)
        with col_left:
            show_figure(FIGURES_DIR / "roc_pr.png", "Discrimination: ROC & PR")
            show_figure(FIGURES_DIR / "decision_curve.png",
                        "Net benefit vs. treat-all / treat-none")
        with col_right:
            show_figure(FIGURES_DIR / "calibration.png",
                        "Calibration (reliability) curve")

    st.divider()
    st.subheader("What drives the predictions? (SHAP)")
    if shap_top is not None:
        agg = shap_top.get("top_features_aggregated", {})
        if agg:
            top5 = list(agg.items())[:5]
            st.write("Top drivers (aggregated to source variables): "
                     + ", ".join(f"**{k}**" for k, _ in top5))
        st.caption(
            "Honest finding: this model leans heavily on discharge disposition "
            "and medical specialty — `discharge_disposition_id` is flagged for "
            "leakage/shortcut scrutiny."
        )
    cshap1, cshap2 = st.columns(2)
    with cshap1:
        show_figure(FIGURES_DIR / "shap_importance_grouped.png",
                    "Global importance (aggregated)")
        show_figure(FIGURES_DIR / "shap_waterfall_high.png",
                    "Local explanation — high-risk patient")
    with cshap2:
        show_figure(FIGURES_DIR / "shap_summary.png", "SHAP beeswarm")
        show_figure(FIGURES_DIR / "shap_waterfall_low.png",
                    "Local explanation — low-risk patient")


# --------------------------------------------------------------------------- #
# Tab 2 — Fairness
# --------------------------------------------------------------------------- #

with tab_fairness:
    st.subheader("Where is the model least reliable?")
    if not need(fairness, "python src/fairness.py"):
        st.caption(
            f"Per-subgroup performance at an outreach operating threshold of "
            f"**{fairness['threshold']:.2f}**. Groups with n < 100 are flagged "
            f"low-evidence and excluded from disparity gaps — the first fairness "
            f"finding is often *insufficient data*."
        )
        for attr, payload in fairness["attributes"].items():
            st.markdown(f"#### By {attr}")
            disp = payload["disparities"]
            d1, d2, d3 = st.columns(3)
            if disp.get("tpr_gap") is not None:
                d1.metric("Recall (TPR) gap", f"{disp['tpr_gap']:.3f}",
                          help=f"min: {disp.get('tpr_min_group')}, "
                               f"max: {disp.get('tpr_max_group')}")
            if disp.get("selection_rate_gap") is not None:
                d2.metric("Selection-rate gap", f"{disp['selection_rate_gap']:.3f}")
            if disp.get("auroc_gap") is not None:
                d3.metric("AUROC gap", f"{disp['auroc_gap']:.3f}")

            df = pd.DataFrame(payload["by_group"])
            show_cols = [c for c in ["group", "count", "selection_rate", "tpr",
                                     "fpr", "precision", "auroc", "mean_pred",
                                     "observed_rate", "low_evidence"]
                         if c in df.columns]
            st.dataframe(df[show_cols], hide_index=True, width="stretch")
            show_figure(FIGURES_DIR / f"fairness_{attr}.png")
            st.divider()


# --------------------------------------------------------------------------- #
# Tab 3 — Monitoring (the hero)
# --------------------------------------------------------------------------- #

with tab_monitor:
    st.subheader("Post-deployment drift monitoring")
    if not need(drift, "python src/drift.py"):
        ref = drift["reference"]
        thr = drift["thresholds"]
        scenarios = drift["scenarios"]

        st.caption(
            "Pick a scenario representing incoming production data. The monitor "
            "compares it against the validated **reference** window and trips a "
            "retraining alert when data drift is widespread or performance "
            "decays past policy thresholds."
        )

        scenario_labels = {
            "baseline": "Baseline (no shift) — control",
            "age_shift": "Age shift — population skews older",
            "pipeline_break": "Pipeline break — a field collapses upstream",
            "prevalence_surge": "Prevalence surge — COVID-like readmission spike",
        }
        choice = st.selectbox(
            "Scenario / monitoring window",
            options=list(scenarios.keys()),
            format_func=lambda k: scenario_labels.get(k, k),
        )
        s = scenarios[choice]

        # The hero banner.
        if s["retrain_recommended"]:
            st.error("### 🚨 RETRAIN RECOMMENDED\n\n" +
                     "\n".join(f"- {r}" for r in s["reasons"]), icon="🚨")
        else:
            st.success("### ✅ Model healthy — no retraining indicated\n\n"
                       "Drift and performance are within policy thresholds.",
                       icon="✅")

        st.markdown("#### Metrics vs. validated reference")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AUROC", f"{s['auroc']:.3f}",
                  delta=f"{s['auroc'] - ref['auroc']:+.3f}")
        # For Brier, lower is better -> invert delta colour.
        m2.metric("Brier", f"{s['brier']:.4f}",
                  delta=f"{s['brier'] - ref['brier']:+.4f}",
                  delta_color="inverse")
        m3.metric("Drift share", f"{s['drift_share']:.0%}",
                  help=f"Alert at >= {thr['drift_share_alert']:.0%} of features")
        m4.metric("Readmit prevalence", f"{s['prevalence']:.1%}",
                  delta=f"{s['prevalence'] - ref['prevalence']:+.1%}",
                  delta_color="off")
        m5.metric("Outreach alert rate", f"{s['alert_rate']:.1%}")

        with st.expander("Alert policy (thresholds)"):
            st.write(
                f"- **Data drift:** retrain if ≥ {thr['drift_share_alert']:.0%} "
                f"of features drift.\n"
                f"- **Discrimination:** retrain if AUROC falls ≥ "
                f"{thr['auroc_drop_alert']} vs. reference.\n"
                f"- **Calibration:** retrain if Brier rises ≥ "
                f"{thr['brier_rise_alert']} vs. reference.\n\n"
                f"Reference window: n = {ref['n']:,}, AUROC = {ref['auroc']:.3f}, "
                f"Brier = {ref['brier']:.4f}, prevalence = {ref['prevalence']:.3f}."
            )

        # Cross-scenario summary table.
        st.markdown("#### All scenarios at a glance")
        rows = []
        for name, sc in scenarios.items():
            rows.append({
                "scenario": name,
                "drift_share": sc["drift_share"],
                "AUROC": sc["auroc"],
                "Brier": sc["brier"],
                "prevalence": sc["prevalence"],
                "retrain?": "🚨 yes" if sc["retrain_recommended"] else "✅ no",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     width="stretch")

        # Embed the full Evidently report on demand (the files are large).
        html_path = REPORTS_DIR / s["html"]
        with st.expander("📄 Full Evidently drift report (interactive)"):
            if html_path.exists():
                # components.html embeds the raw interactive report via an
                # iframe srcdoc; st.iframe only takes a URL, so it can't render
                # this in-memory HTML directly.
                components.html(html_path.read_text(encoding="utf-8"),
                                height=600, scrolling=True)
            else:
                st.info(f"`{s['html']}` not found — run `python src/drift.py` "
                        "to regenerate the Evidently reports.")

st.divider()
st.caption(
    "Research/portfolio demonstration on the public UCI Diabetes 130-US "
    "Hospitals dataset — not a validated clinical tool. Decision *support* for "
    "care-management outreach only; must never gate coverage or access to care."
)
