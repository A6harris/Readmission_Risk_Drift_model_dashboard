# Phase 5 — Explainability (SHAP)

## Global drivers

Aggregated to source clinical variables (one-hot columns summed back together),
the strongest drivers of predicted 30-day readmission risk are:

1. **discharge_disposition_id** (summed mean |SHAP| = 3.070)
2. **medical_specialty** (summed mean |SHAP| = 2.937)
3. **diag_1_group** (summed mean |SHAP| = 0.906)
4. **admission_source_id** (summed mean |SHAP| = 0.845)
5. **diag_2_group** (summed mean |SHAP| = 0.838)
6. **diag_3_group** (summed mean |SHAP| = 0.534)
7. **repaglinide** (summed mean |SHAP| = 0.375)
8. **admission_type_id** (summed mean |SHAP| = 0.366)
9. **insulin** (summed mean |SHAP| = 0.237)
10. **age** (summed mean |SHAP| = 0.190)
11. **metformin** (summed mean |SHAP| = 0.185)
12. **number_inpatient** (summed mean |SHAP| = 0.179)
13. **A1Cresult** (summed mean |SHAP| = 0.160)
14. **race** (summed mean |SHAP| = 0.155)
15. **num_lab_procedures** (summed mean |SHAP| = 0.127)

The strongest *individual* encoded columns (before aggregation) are:

1. `discharge_disposition_id_23` (0.884)
2. `medical_specialty_Hematology/Oncology` (0.661)
3. `discharge_disposition_id_28` (0.647)
4. `repaglinide_Up` (0.366)
5. `admission_source_id_3` (0.323)
6. `discharge_disposition_id_25` (0.320)
7. `discharge_disposition_id_22` (0.315)
8. `discharge_disposition_id_2` (0.292)
9. `diag_2_group_Neoplasms` (0.266)
10. `medical_specialty_Orthopedics-Reconstructive` (0.247)

See `figures/shap_summary.png` for the per-feature beeswarm and
`figures/shap_importance_grouped.png` for the aggregated ranking above. Summing
absolute SHAP over one-hot columns does favor higher-cardinality variables, but
here the per-column ranking tells the same story — specific discharge
dispositions and specialties dominate on their own, so this is real reliance,
not just an aggregation artifact.

### Are these clinically plausible?

- **`discharge_disposition_id`** — where the patient is sent after discharge (home, skilled nursing, transfer). Known at discharge time, so it is legitimately available — but the model's heavy reliance on specific administrative codes is exactly the kind of shortcut behavior worth auditing for leakage.
- **`medical_specialty`** — the attending service, a proxy for case mix and acuity (e.g. oncology vs. orthopedics carry very different baseline risk).
- **`diag_1_group`** — primary diagnosis category — direct clinical signal.
- **`admission_source_id`** — how the patient arrived (ER, physician referral, transfer) — clinically meaningful context for acuity.
- **`diag_2_group`** — secondary diagnosis category — comorbidity signal.
- **`diag_3_group`** — additional diagnosis category — comorbidity signal.
- **`repaglinide`** — repaglinide use/changes — relevant in this diabetic cohort.
- **`admission_type_id`** — admission type (emergency vs. elective), a reasonable acuity proxy.
- **`insulin`** — insulin use/changes — relevant in this diabetic cohort.
- **`age`** — age, expected to matter — but also the variable with the widest performance gap in the fairness audit (Phase 4), so read its influence alongside that caveat.
- **`metformin`** — metformin use/changes — relevant in this diabetic cohort.
- **`number_inpatient`** — prior inpatient admissions — the classic, well-validated readmission predictor.

Notably, raw prior-utilization counts (`number_inpatient`, `number_emergency`, `number_outpatient`) rank *lower* than the readmission literature would lead you to expect. This model leans more on administrative/categorical signals (discharge disposition, medical specialty, diagnosis groups) than on prior-utilization tallies — an honest finding that itself warrants scrutiny before any deployment.

"Plausible" is not "validated": several top drivers are administrative proxies
available at discharge, and the per-prediction explanations below are what a
clinician would actually inspect before acting on an alert.

## Local explanations

Two individual predictions illustrate how the same drivers play out per patient
(`figures/shap_waterfall_high.png`, `figures/shap_waterfall_low.png`):

- **High-risk example** — predicted probability 0.625
  (actual outcome: 1). The waterfall shows which features
  push this patient's risk above the base rate.
- **Low-risk example** — predicted probability 0.015
  (actual outcome: 0). For this patient the same kinds of
  features push the prediction the other way.

Each waterfall starts from the model's base rate and shows the additive
contribution of each feature, in log-odds — the transparency a care team needs
to trust (or override) an individual alert.
