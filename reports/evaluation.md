# Phase 3 — Model Evaluation

_Held-out test set (n = 13,998), 30-day readmission prevalence = 0.090._
Serialized model: **xgboost**.

## Discrimination
- **AUROC = 0.658** — modest, and honestly so. 30-day readmission is a
  genuinely hard outcome to predict from administrative data; published models
  on this dataset sit in a similar range. A respectable-looking AUROC would be
  a red flag, not a triumph.
- **AUPRC = 0.188** vs. a no-skill baseline of 0.090. The model
  carries real signal over chance, but precision is inherently limited by the
  low base rate — most flagged patients will not be readmitted within 30 days.

## Calibration
- **Brier score = 0.0785.** The reliability curve (`figures/calibration.png`)
  shows how closely predicted risks track observed rates. Calibration matters
  more than discrimination here: outreach capacity is allocated against the
  *probabilities*, so systematically over- or under-stated risk wastes scarce
  care-management time.

## Net benefit (decision-curve analysis)
- Across threshold probabilities in roughly **4%–49%**,
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
