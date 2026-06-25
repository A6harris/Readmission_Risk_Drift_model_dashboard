# PLAN.md — Readmission Risk Model + Drift Monitoring Dashboard

> A healthcare ML portfolio project that demonstrates not just modeling, but **responsible deployment**: subgroup fairness auditing, explainability, and continuous post-deployment monitoring for dataset shift. Built to reflect how health systems actually govern AI — "validate locally, monitor continuously."

**Author:** _[your name]_
**Status:** Planning
**Est. timeline:** Focused weekend + a few evenings of polish

---

## 1. The pitch (put a version of this at the top of your README)

Most healthcare ML demos stop at "I trained a model and got X AUC." Real clinical AI fails in a more interesting place: **after** deployment, when the population shifts, the data pipeline changes, or the model meets a hospital it wasn't trained on — and nobody notices until patients are affected. This project builds a 30-day hospital readmission risk model, then surrounds it with the governance layer that production healthcare AI requires: a fairness audit across demographic subgroups, SHAP explanations, calibration and net-benefit analysis, and a live **monitoring dashboard** that detects data/model drift over time and raises retraining alerts.

The point isn't the AUC. The point is everything around it.

---

## 2. Why this matters (and the research it's based on)

Each design decision below maps to a specific finding in the clinical-AI literature. These aren't decoration — cite them in your README's "Background" section; it's what makes this read like the work of someone who understands healthcare AI, not just scikit-learn.

- **Why monitor for drift at all** → Models silently degrade when the deployment distribution diverges from training. This is the core argument of Finlayson et al., *The Clinician and Dataset Shift in Artificial Intelligence* (NEJM, 2021) [Ref 1]. The COVID-19 era is a vivid natural example: Wong et al. showed a deployed sepsis model's alert behavior shifted measurably during the pandemic [Ref 6].
- **Why post-deployment surveillance is its own discipline** → Ansari, Baur, Singh & Admon, *Challenges in the Postmarket Surveillance of Clinical Prediction Models* (NEJM AI, 2025) [Ref 2] argues monitoring is under-specified and under-built — exactly the gap this dashboard fills. Operationalized further in the Stanford *Monitoring Deployed AI Systems in Health Care* framework [Ref 8].
- **Why audit performance across subgroups/sites** → A model's headline metric hides wide variation across populations and institutions. Lyons et al. found a single proprietary model performed very differently across 9 networked hospitals [Ref 4]; the broader external-validation lesson comes from Wong et al.'s landmark finding that a widely deployed sepsis model worked far worse than advertised [Ref 3]. The fairness audit is the local-validation principle applied to demographic subgroups.
- **Why include net-benefit / decision-curve analysis, not just AUC** → A model's value depends on whether you can act on its alerts given real resource constraints. See Singh, Shah & Vickers, *Assessing the net benefit of machine learning models in the presence of resource constraints* (JAMIA, 2023) [Ref 5], building on Vickers' decision-curve analysis [Ref 9].
- **Why write a model card to a standard** → Use the TRIPOD+AI reporting checklist [Ref 6/Ref 7] and PROBAST+AI risk-of-bias framework [Ref 10] to structure the model card. Referencing these signals you know the field's reporting norms.
- **Governance framing** → Situate the whole thing against the national debate on AI oversight infrastructure, e.g. Shah et al., *Nationwide Network of Health AI Assurance Laboratories* (JAMA, 2024) [Ref 11].

---

## 3. Tech stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | Python 3.11 | |
| Data / modeling | pandas, scikit-learn, XGBoost (or LightGBM) | Tabular, small — fast iteration |
| Explainability | SHAP | Global + per-prediction |
| Fairness | Fairlearn (or aequitas) | Subgroup metrics, disparity dashboards |
| Calibration / net benefit | sklearn calibration, custom decision-curve fn | |
| Drift detection | Evidently | Data drift + model performance reports |
| Dashboard | Streamlit | Photographs well → README GIF |
| Plots | plotly / matplotlib | |
| Reproducibility | requirements.txt + Dockerfile | One-command run |
| (Stretch) Orchestration | a simple `make` or Python CLI | Simulate "monitoring run" on a schedule |

---

## 4. Data

**Dataset:** "Diabetes 130-US Hospitals for Years 1999–2008" (UCI Machine Learning Repository).
- ~101,766 inpatient encounters; demographics (race, gender, age), diagnoses, medications, labs, prior utilization.
- **Target:** readmission within 30 days (binary: `<30` vs `not <30`) — collapse the original 3-class label.
- **Origin paper:** Strack et al., 2014 [Ref 12] — cite this for dataset provenance.
- **Why this dataset:** open (no credentialing), clean enough to move fast, and crucially it contains `race`, `gender`, and `age` columns, which makes the **fairness audit real** rather than hypothetical.
- **Known gotchas (let Claude Code handle these):** missing values coded as `?`; `weight` and `payer_code` are mostly missing (drop or treat carefully); multiple encounters per patient (decide whether to de-duplicate to avoid leakage); diagnosis codes are ICD-9 and need grouping.

> **Note on synthetic vs. real data:** state clearly in the README that this is a public research dataset used for demonstration. That judgment is itself a positive signal to a payer.

---

## 5. Repo structure

```
readmission-monitoring/
├── README.md                 # pitch + background (cite the papers) + screenshots/GIF
├── PLAN.md                   # this file
├── requirements.txt
├── Dockerfile
├── data/
│   ├── raw/                  # downloaded UCI data (gitignored)
│   └── processed/
├── src/
│   ├── data_prep.py          # load, clean, encode, split
│   ├── train.py              # train + serialize model
│   ├── evaluate.py           # AUC, calibration, decision curve / net benefit
│   ├── fairness.py           # subgroup metrics via Fairlearn
│   ├── explain.py            # SHAP global + local
│   ├── drift.py              # simulate shift + Evidently reports
│   └── monitor_app.py        # Streamlit dashboard
├── models/                   # serialized model + model_card.md
├── reports/                  # generated Evidently HTML, figures
├── notebooks/                # optional EDA scratchpad
└── tests/                    # a few pytest sanity checks
```

---

## 6. Build sequence (Claude Code–sized steps)

Work phase by phase; each is a clean, self-contained prompt to Claude Code. Commit after each.

**Phase 0 — Scaffold.** Create the repo structure above, `requirements.txt`, a `.gitignore` (ignore `data/raw`, models, `__pycache__`), and a stub README. Set up a virtual env.

**Phase 1 — Data prep (`data_prep.py`).** Download the UCI dataset, handle `?` missing values, drop/repair the mostly-empty columns, collapse the target to binary `readmitted_<30`, group ICD-9 diagnosis codes into clinical categories, encode categoricals, and produce a reproducible train/test split (stratified). De-duplicate patients to prevent leakage. Output to `data/processed/`.

**Phase 2 — Baseline model (`train.py`).** Train a regularized logistic regression as an interpretable baseline, then XGBoost/LightGBM as the main model. Serialize the best model + the preprocessing pipeline together. Keep it simple; resist over-tuning.

**Phase 3 — Evaluation (`evaluate.py`).** Report AUROC, AUPRC, a calibration curve (Brier score), and a **decision-curve / net-benefit plot** [Ref 5, 9]. Write a short interpretation. This is where you show you evaluate like a clinician, not a Kaggler.

**Phase 4 — Fairness audit (`fairness.py`).** Using Fairlearn, compute per-subgroup performance (by `race`, `gender`, `age` band): selection rate, TPR/FPR, calibration. Produce disparity plots and a written "where this model is least reliable" section [Ref 3, 4].

**Phase 5 — Explainability (`explain.py`).** SHAP summary plot (global drivers) + a couple of individual-prediction force plots. Comment on whether the drivers are clinically plausible.

**Phase 6 — Drift simulation + detection (`drift.py`).** Create 2–3 "shifted" versions of the test set (e.g., age-distribution shift, a feature-pipeline change that drops/renames a field, a label-prevalence shift mimicking a COVID-like event [Ref 1, 6]). Run Evidently to generate data-drift and model-performance reports for each. Save HTML to `reports/`.

**Phase 7 — Monitoring dashboard (`monitor_app.py`).** Streamlit app with: a model-overview tab (metrics, calibration, SHAP), a fairness tab, and a **monitoring tab** where the user selects a time window / scenario and sees drift metrics, performance decay, and a clear **"RETRAIN RECOMMENDED" alert** when thresholds trip. This is your hero screenshot.

**Phase 8 — Model card + README + reproducibility.** Generate `models/model_card.md` following TRIPOD+AI structure [Ref 7]. Write the real README (pitch, background with citations, screenshots/GIF, how-to-run, responsible-AI section, limitations). Add the Dockerfile so `docker build && docker run` launches the dashboard. A few pytest tests in `tests/`.

**Phase 9 — Stretch (optional).** Deploy the dashboard (Streamlit Community Cloud / Hugging Face Spaces) for a clickable live link; add a tiny "retraining trigger" script that re-fits when drift exceeds a threshold and logs the event.

---

## 7. Responsible-AI section (include this in the README)

Structure it as: **Intended use** (decision *support* for care-management outreach prioritization, not autonomous denial of anything) · **Out-of-scope uses** (must never gate coverage or access to care) · **Subgroups where performance is weakest** (from Phase 4) · **What would need to be true to deploy** (prospective validation on local data, monitoring in place, human in the loop) · **Failure modes** (dataset shift, feedback loops). This section is the single most differentiating part of the repo for a payer audience.

---

## 8. Paste-ready Claude Code starter prompt

```
I'm building a healthcare ML portfolio project: a 30-day hospital readmission
risk model wrapped in a responsible-deployment layer (fairness audit, SHAP,
calibration + net-benefit analysis) and a Streamlit dashboard that detects
dataset drift over time and raises retraining alerts.

Dataset: UCI "Diabetes 130-US Hospitals for Years 1999-2008".
Stack: Python 3.11, scikit-learn + XGBoost, SHAP, Fairlearn, Evidently,
Streamlit, Docker.

Start with Phase 0 + Phase 1 from my PLAN.md (in the repo root): scaffold the
repo structure, requirements.txt, .gitignore, and a stub README, then write
src/data_prep.py to download and clean the dataset, handle '?' missing values,
collapse the target to binary readmitted-within-30-days, group ICD-9 diagnoses,
encode categoricals, de-duplicate patients to prevent leakage, and produce a
stratified train/test split saved to data/processed/.

Read PLAN.md first and follow the phase structure. Stop after Phase 1 so I can
review before we train.
```

---

## 9. References

1. Finlayson SG, Subbaswamy A, Singh K, Bowers J, Kupke A, Zittrain J, Kohane IS, Saria S. **The Clinician and Dataset Shift in Artificial Intelligence.** *N Engl J Med.* 2021 Jul 15;385(3):283-286. PMID: 34260843. https://pubmed.ncbi.nlm.nih.gov/34260843
2. Ansari S, Baur B, Singh K, Admon AJ. **Challenges in the Postmarket Surveillance of Clinical Prediction Models.** *NEJM AI.* 2025 May;2(5). PMID: 40873499. https://pubmed.ncbi.nlm.nih.gov/40873499
3. Wong A, Otles E, Donnelly JP, Krumm A, McCullough J, DeTroyer-Cooley O, Pestrue J, Phillips M, Konye J, Penoza C, Ghous M, Singh K. **External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients.** *JAMA Intern Med.* 2021 Aug 1;181(8):1065-1070. PMID: 34152373. https://pubmed.ncbi.nlm.nih.gov/34152373
4. Lyons PG, Hofford MR, Yu SC, Michelson AP, Payne PRO, Hough CL, Singh K. **Factors Associated With Variability in the Performance of a Proprietary Sepsis Prediction Model Across 9 Networked Hospitals in the US.** *JAMA Intern Med.* 2023 Jun 1;183(6):611-612. PMID: 37010858. https://pubmed.ncbi.nlm.nih.gov/37010858
5. Singh K, Shah NH, Vickers AJ. **Assessing the net benefit of machine learning models in the presence of resource constraints.** *J Am Med Inform Assoc.* 2023 Mar 16;30(4):668-673. PMID: 36810659. https://pubmed.ncbi.nlm.nih.gov/36810659
6. Wong A, Cao J, Lyons PG, Dutta S, Major VJ, Ötles E, Singh K. **Quantification of Sepsis Model Alerts in 24 US Hospitals Before and During the COVID-19 Pandemic.** *JAMA Netw Open.* 2021 Nov 1;4(11):e2135286. PMID: 34797372. https://pubmed.ncbi.nlm.nih.gov/34797372
7. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. **TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods.** *BMJ.* 2024 Apr 16;385:e078378. PMID: 38626948. https://pubmed.ncbi.nlm.nih.gov/38626948
8. **Monitoring Deployed AI Systems in Health Care** (Stanford Health Care; Responsible AI Lifecycle / RAIL framework). 2025. arXiv:2512.09048. https://arxiv.org/abs/2512.09048
9. Vickers AJ, Elkin EB. **Decision curve analysis: a novel method for evaluating prediction models.** *Med Decis Making.* 2006;26(6):565-574. PMID: 17099194. https://pubmed.ncbi.nlm.nih.gov/17099194
10. Moons KGM, Damen JAA, Kaul T, Hooft L, et al. **PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods.** *BMJ.* 2025 Mar 24;388:e082505. PMID: 40127903. https://pubmed.ncbi.nlm.nih.gov/40127903
11. Shah NH, Halamka JD, Saria S, et al. **A Nationwide Network of Health AI Assurance Laboratories.** *JAMA.* 2024;331(3):245-249. https://jamanetwork.com/journals/jama/fullarticle/2813425
12. Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, Clore JN. **Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.** *BioMed Research International.* 2014;2014:781670. (Origin of the UCI Diabetes 130-US Hospitals dataset.) https://doi.org/10.1155/2014/781670

---

*Tip: the README's "Background" paragraph + References list is what makes a reviewer pause. Lead the README with the pitch and a dashboard GIF, then the background-with-citations, then how-to-run. Make them feel the governance story before they read a line of code.*