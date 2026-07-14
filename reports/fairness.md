# Phase 4 — Fairness Audit

Per-subgroup performance on the held-out test set, at an outreach operating threshold of **0.10** (risk ≥ threshold → flag).

> Fairness here means *reliability across groups*, not a single pass/fail number. Subgroups with n < 100 are reported but flagged as low-evidence and excluded from disparity gaps.

## By race

| group | count | selection_rate | tpr | fpr | precision | auroc | mean_pred | observed_rate |
|---|---|---|---|---|---|---|---|---|
| AfricanAmerican | 2504 | 0.256 | 0.481 | 0.233 | 0.173 | 0.679 | 0.084 | 0.092 |
| Asian | 96 | 0.198 | 0.444 | 0.172 | 0.211 | 0.796 | 0.076 | 0.094 |  ⚠️
| Caucasian | 10522 | 0.314 | 0.499 | 0.295 | 0.144 | 0.648 | 0.091 | 0.091 |
| Hispanic | 286 | 0.269 | 0.533 | 0.238 | 0.208 | 0.697 | 0.083 | 0.105 |
| Missing | 361 | 0.213 | 0.500 | 0.195 | 0.143 | 0.702 | 0.076 | 0.061 |
| Other | 229 | 0.197 | 0.545 | 0.179 | 0.133 | 0.818 | 0.075 | 0.048 |

- **Recall (TPR) gap:** 0.065 — lowest for `AfricanAmerican`, highest for `Other`. A lower TPR means the model *misses more* true readmissions in that group.
- **Selection-rate gap:** 0.117 — groups are flagged for outreach at different rates.
- **AUROC gap:** 0.170 across reliable groups.

## By gender

| group | count | selection_rate | tpr | fpr | precision | auroc | mean_pred | observed_rate |
|---|---|---|---|---|---|---|---|---|
| Female | 7507 | 0.313 | 0.527 | 0.292 | 0.149 | 0.662 | 0.090 | 0.088 |
| Male | 6491 | 0.279 | 0.462 | 0.260 | 0.151 | 0.655 | 0.088 | 0.091 |

- **Recall (TPR) gap:** 0.065 — lowest for `Male`, highest for `Female`. A lower TPR means the model *misses more* true readmissions in that group.
- **Selection-rate gap:** 0.034 — groups are flagged for outreach at different rates.
- **AUROC gap:** 0.007 across reliable groups.

## By age

| group | count | selection_rate | tpr | fpr | precision | auroc | mean_pred | observed_rate |
|---|---|---|---|---|---|---|---|---|
| [0-10) | 29 | 0.034 | 0.000 | 0.034 | 0.000 | n/a | 0.039 | 0.000 |  ⚠️
| [10-20) | 109 | 0.083 | 0.400 | 0.067 | 0.222 | 0.804 | 0.055 | 0.046 |
| [20-30) | 204 | 0.167 | 0.667 | 0.127 | 0.294 | 0.764 | 0.073 | 0.074 |
| [30-40) | 537 | 0.151 | 0.314 | 0.139 | 0.136 | 0.712 | 0.069 | 0.065 |
| [40-50) | 1352 | 0.174 | 0.330 | 0.162 | 0.132 | 0.627 | 0.073 | 0.070 |
| [50-60) | 2466 | 0.147 | 0.333 | 0.131 | 0.180 | 0.671 | 0.069 | 0.079 |
| [60-70) | 3125 | 0.277 | 0.409 | 0.264 | 0.129 | 0.623 | 0.089 | 0.088 |
| [70-80) | 3606 | 0.389 | 0.611 | 0.362 | 0.165 | 0.662 | 0.102 | 0.105 |
| [80-90) | 2239 | 0.469 | 0.640 | 0.450 | 0.139 | 0.646 | 0.108 | 0.102 |
| [90-100) | 331 | 0.360 | 0.485 | 0.346 | 0.134 | 0.628 | 0.099 | 0.100 |

- **Recall (TPR) gap:** 0.352 — lowest for `[30-40)`, highest for `[20-30)`. A lower TPR means the model *misses more* true readmissions in that group.
- **Selection-rate gap:** 0.387 — groups are flagged for outreach at different rates.
- **AUROC gap:** 0.181 across reliable groups.

## Where this model is least reliable

Read the gaps above as a deployment caveat, not a verdict. The honest takeaways for a care team would be:

- Estimates for small subgroups (flagged ⚠️) are too noisy to act on — the first fairness finding is often *insufficient data*, which is itself a reason not to deploy blindly.
- Any subgroup with materially lower recall is one where outreach would systematically under-serve real readmissions; that group needs either a group-specific threshold or a human-in-the-loop safeguard before use.
- These results are on a historical (1999–2008) dataset and would need to be re-checked on local, contemporary data before any deployment.
