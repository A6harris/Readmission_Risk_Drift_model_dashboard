# Model Card — 30-Day Hospital Readmission Risk

This card follows the spirit of the **TRIPOD+AI** reporting checklist (Collins
et al., *BMJ* 2024) and is informed by the **PROBAST+AI** risk-of-bias framework
(Moons et al., *BMJ* 2025). It documents a research/portfolio model, not a
deployable clinical tool.

> **TL;DR.** A calibrated gradient-boosted model estimates the probability that a
> hospital encounter results in readmission within 30 days. Discrimination is
> modest (AUROC ≈ 0.66) — as expected for this outcome — but the model is well
> calibrated and adds net benefit across a realistic outreach threshold band.
> It is intended only as decision *support* for prioritizing care-management
> outreach, and must never be used to gate coverage or access to care.

---

## 1. Model details

| Field | Value |
|---|---|
| Model type | XGBoost gradient-boosted trees (selected over an L2 logistic-regression baseline) |
| Selection | 5-fold stratified CV on the training set, by mean AUPRC |
| Preprocessing | `StandardScaler` (numeric) + `OneHotEncoder(handle_unknown="ignore")` (categorical), bundled in one scikit-learn `Pipeline` and serialized with the model |
| Output | Calibrated probability of `readmitted_lt30` (1 = readmitted < 30 days) |
| Class imbalance | **Not** rebalanced — see §6; handled at decision time via the operating threshold |
| Version / artifact | `models/model.joblib` (pipeline + feature spec) |
| Reproducibility | `requirements.txt`, fixed `random_state=42`, one-command pipeline + Dockerfile |

## 2. Intended use

- **Intended use.** Decision *support* for prioritizing care-management outreach
  — helping a care team decide whom to check in on after discharge.
- **Intended users.** Care-management / population-health teams, with a human
  clinician in the loop for any individual decision.
- **Out-of-scope / prohibited.** Must **never** gate, deny, or delay coverage,
  benefits, or access to care. Not validated for autonomous decision-making, for
  individual diagnosis, or for any population other than the one described below.

## 3. Data

- **Source.** UCI "Diabetes 130-US Hospitals for Years 1999–2008"
  (Strack et al., 2014). Public research dataset — **not** real member/patient
  data.
- **Cohort construction (leakage-controlled).**
  - Decoded `?` sentinels to missing.
  - Removed encounters discharged to hospice or expired (cannot be readmitted →
    outcome leakage).
  - De-duplicated to **one encounter per patient** (prevents the same patient
    appearing in both train and test).
  - Dropped mostly-empty/non-predictive columns (`weight`, `payer_code`) and
    zero-variance medication columns.
  - Grouped ICD-9 `diag_1/2/3` into clinical categories; engineered an
    `age_midpoint` from age bands; preserved "lab not measured" as an explicit
    category.
- **Final cohort.** 69,987 unique-patient encounters; 43 features
  (9 numeric, 34 categorical). **30-day readmission prevalence ≈ 9.0%.**
- **Split.** Stratified 80/20 train/test (55,989 / 13,998), `random_state=42`.

## 4. Performance (held-out test set, n = 13,998)

| Metric | Value | Note |
|---|---|---|
| AUROC | **0.658** | Modest by design; 30-day readmission is hard from administrative data |
| AUPRC | **0.187** | vs. a no-skill baseline of 0.090 (the prevalence) |
| Brier score | **0.0785** | Beats the no-skill Brier of 0.0817 → genuinely calibrated |
| Net benefit | Positive across ~**3%–51%** thresholds | Beats treat-all / treat-none in the actionable band |

Baseline comparison (test): logistic regression AUROC 0.651 / AUPRC 0.173.
The gradient-boosted model wins modestly. See `reports/evaluation.md`.

## 5. Fairness / subgroup reliability

Audited by `race`, `gender`, and `age` at an outreach threshold of 0.10
(`reports/fairness.md`). Calibration holds across groups (predicted ≈ observed
rate). Notable disparities among subgroups with n ≥ 100:

- **Age** is where reliability varies most: recall (TPR) gap ≈ **0.35** (lowest
  in `[30-40)`, highest in `[20-30)`), AUROC gap ≈ 0.18, and selection rate
  climbs steeply with age — older patients are flagged far more often.
- **Race**: recall gap ≈ 0.18 among reliable groups; the smallest groups (e.g.
  Asian) are **too small (n < 100) to draw conclusions** — itself a key finding.
- **Gender**: differences are modest (recall gap ≈ 0.07).

**Weakest where:** small subgroups (insufficient data) and the working-age
`[30-40)` band (lowest recall) — outreach there would under-serve real
readmissions without a group-specific threshold or human review.

## 6. Key modeling decisions & limitations

- **No class rebalancing — deliberate.** `class_weight='balanced'` /
  `scale_pos_weight` improved ranking marginally but inflated probabilities,
  producing a Brier (0.215) *worse than predicting the base rate*. Since this
  project's thesis is that calibration matters, the model is trained on natural
  proportions and imbalance is handled at decision time.
- **The model leans on administrative signals.** SHAP shows the top drivers are
  `discharge_disposition_id` and `medical_specialty` (at both aggregated and
  per-column level), while raw prior-utilization counts rank lower than the
  literature expects. `discharge_disposition_id` is **flagged for leakage /
  shortcut scrutiny** (`reports/explainability.md`).
- **Historical, single-source data (1999–2008).** Not representative of any
  current population; no external/temporal validation.
- **Not prospectively validated**, and the outcome label is administrative.

## 7. Monitoring & maintenance

Post-deployment drift is simulated and detected in `src/drift.py` /
`reports/drift_summary.json`, surfaced in the Streamlit dashboard. Each
scenario is scored over a stream of monitoring windows in which the shift
ramps up mid-stream. Per window, vs. a validated reference window, the alert
policy is:

- **RETRAIN tier** — ≥ 30% of features drift (Evidently), **or** AUROC falls
  ≥ 0.03, **or** Brier rises ≥ 0.02;
- **WARNING tier** — any of those metrics ≥ 50% of the way to its threshold;
- **Sustained-breach rule** — retraining is recommended only after ≥ 2
  consecutive RETRAIN-tier windows, so a single noisy window cannot trigger a
  retrain (it surfaces as WARNING instead).

Alerts are attributable: the dashboard lists per-feature drift scores for the
latest window, and every decision by `src/retrain_trigger.py` is appended to
an audit log (`models/retrain_log.jsonl`).

Demonstrated behavior: a **pipeline break** (a field collapsing upstream)
trips the AUROC rule once enough rows are affected; a **prevalence surge**
(COVID-like) trips the Brier rule with essentially *zero* feature drift —
showing that label shift is invisible to feature-drift monitoring and must be
caught by performance tracking. A benign **age shift** is detected and held at
WARNING but correctly never triggers retraining.

## 8. What would need to be true to deploy responsibly

1. Prospective validation on **local, contemporary** data.
2. Subgroup performance within agreed bounds (and enough data per subgroup).
3. Resolution of the `discharge_disposition_id` reliance (leakage audit).
4. An active drift-monitoring process (the kind prototyped here).
5. A human clinician in the loop for every individual decision.

## References

Collins et al., *TRIPOD+AI*, BMJ 2024 · Moons et al., *PROBAST+AI*, BMJ 2025 ·
Strack et al., BMC 2014 · Vickers & Elkin, *Decision curve analysis*, 2006 ·
Singh, Shah & Vickers, JAMIA 2023 · Finlayson et al., NEJM 2021. Full list in
the repository README.
