# Phase 4 — Fairness Audit

Per-subgroup performance on the held-out test set, at an outreach operating threshold of **0.10** (risk ≥ threshold → flag).

> Fairness here means *reliability across groups*, not a single pass/fail number. Subgroups with n < 100 are reported but flagged as low-evidence and excluded from disparity gaps.

## By race

| group | count | selection_rate | tpr | fpr | precision | auroc | mean_pred | observed_rate |
|---|---|---|---|---|---|---|---|---|
| AfricanAmerican | 2504 | 0.250 | 0.494 | 0.225 | 0.182 | 0.684 | 0.084 | 0.092 |
| Asian | 96 | 0.177 | 0.556 | 0.138 | 0.294 | 0.808 | 0.077 | 0.094 |  ⚠️
| Caucasian | 10522 | 0.311 | 0.496 | 0.292 | 0.145 | 0.647 | 0.092 | 0.091 |
| Hispanic | 286 | 0.248 | 0.500 | 0.219 | 0.211 | 0.686 | 0.082 | 0.105 |
| Missing | 361 | 0.216 | 0.409 | 0.204 | 0.115 | 0.685 | 0.077 | 0.061 |
| Other | 229 | 0.188 | 0.545 | 0.170 | 0.140 | 0.809 | 0.073 | 0.048 |

- **Recall (TPR) gap:** 0.136 — lowest for `Missing`, highest for `Other`. A lower TPR means the model *misses more* true readmissions in that group.
- **Selection-rate gap:** 0.123 — groups are flagged for outreach at different rates.
- **AUROC gap:** 0.163 across reliable groups.

## By gender

| group | count | selection_rate | tpr | fpr | precision | auroc | mean_pred | observed_rate |
|---|---|---|---|---|---|---|---|---|
| Female | 7507 | 0.307 | 0.526 | 0.286 | 0.151 | 0.664 | 0.090 | 0.088 |
| Male | 6491 | 0.277 | 0.460 | 0.258 | 0.152 | 0.651 | 0.089 | 0.091 |

- **Recall (TPR) gap:** 0.065 — lowest for `Male`, highest for `Female`. A lower TPR means the model *misses more* true readmissions in that group.
- **Selection-rate gap:** 0.030 — groups are flagged for outreach at different rates.
- **AUROC gap:** 0.014 across reliable groups.

## By age

| group | count | selection_rate | tpr | fpr | precision | auroc | mean_pred | observed_rate |
|---|---|---|---|---|---|---|---|---|
| [0-10) | 29 | 0.034 | 0.000 | 0.034 | 0.000 | n/a | 0.039 | 0.000 |  ⚠️
| [10-20) | 109 | 0.073 | 0.400 | 0.058 | 0.250 | 0.800 | 0.055 | 0.046 |
| [20-30) | 204 | 0.172 | 0.667 | 0.132 | 0.286 | 0.768 | 0.076 | 0.074 |
| [30-40) | 537 | 0.127 | 0.343 | 0.112 | 0.176 | 0.709 | 0.070 | 0.065 |
| [40-50) | 1352 | 0.167 | 0.319 | 0.156 | 0.133 | 0.620 | 0.073 | 0.070 |
| [50-60) | 2466 | 0.144 | 0.344 | 0.126 | 0.189 | 0.667 | 0.069 | 0.079 |
| [60-70) | 3125 | 0.279 | 0.401 | 0.267 | 0.126 | 0.624 | 0.089 | 0.088 |
| [70-80) | 3606 | 0.385 | 0.611 | 0.358 | 0.166 | 0.663 | 0.102 | 0.105 |
| [80-90) | 2239 | 0.466 | 0.645 | 0.446 | 0.141 | 0.645 | 0.109 | 0.102 |
| [90-100) | 331 | 0.326 | 0.394 | 0.319 | 0.120 | 0.607 | 0.098 | 0.100 |

- **Recall (TPR) gap:** 0.348 — lowest for `[40-50)`, highest for `[20-30)`. A lower TPR means the model *misses more* true readmissions in that group.
- **Selection-rate gap:** 0.392 — groups are flagged for outreach at different rates.
- **AUROC gap:** 0.193 across reliable groups.

## Where this model is least reliable

Read the gaps above as a deployment caveat, not a verdict. The honest takeaways for a care team would be:

- Estimates for small subgroups (flagged ⚠️) are too noisy to act on — the first fairness finding is often *insufficient data*, which is itself a reason not to deploy blindly.
- Any subgroup with materially lower recall is one where outreach would systematically under-serve real readmissions; that group needs either a group-specific threshold or a human-in-the-loop safeguard before use.
- These results are on a historical (1999–2008) dataset and would need to be re-checked on local, contemporary data before any deployment.
