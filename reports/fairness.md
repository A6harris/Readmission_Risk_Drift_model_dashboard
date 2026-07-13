# Phase 4 — Fairness Audit

Per-subgroup performance on the held-out test set, at an outreach operating threshold of **0.10** (risk ≥ threshold → flag).

> Fairness here means *reliability across groups*, not a single pass/fail number. Subgroups with n < 100 are reported but flagged as low-evidence and excluded from disparity gaps.

## By race

| group | count | selection_rate | tpr | fpr | precision | auroc | mean_pred | observed_rate |
|---|---|---|---|---|---|---|---|---|
| AfricanAmerican | 2504 | 0.254 | 0.498 | 0.229 | 0.181 | 0.676 | 0.084 | 0.092 |
| Asian | 96 | 0.146 | 0.333 | 0.126 | 0.214 | 0.797 | 0.075 | 0.094 |  ⚠️
| Caucasian | 10522 | 0.307 | 0.504 | 0.288 | 0.149 | 0.648 | 0.091 | 0.091 |
| Hispanic | 286 | 0.266 | 0.533 | 0.234 | 0.211 | 0.678 | 0.083 | 0.105 |
| Missing | 361 | 0.213 | 0.364 | 0.204 | 0.104 | 0.695 | 0.077 | 0.061 |
| Other | 229 | 0.183 | 0.545 | 0.165 | 0.143 | 0.818 | 0.074 | 0.048 |

- **Recall (TPR) gap:** 0.182 — lowest for `Missing`, highest for `Other`. A lower TPR means the model *misses more* true readmissions in that group.
- **Selection-rate gap:** 0.124 — groups are flagged for outreach at different rates.
- **AUROC gap:** 0.170 across reliable groups.

## By gender

| group | count | selection_rate | tpr | fpr | precision | auroc | mean_pred | observed_rate |
|---|---|---|---|---|---|---|---|---|
| Female | 7507 | 0.308 | 0.535 | 0.286 | 0.154 | 0.664 | 0.090 | 0.088 |
| Male | 6491 | 0.272 | 0.462 | 0.253 | 0.155 | 0.651 | 0.088 | 0.091 |

- **Recall (TPR) gap:** 0.073 — lowest for `Male`, highest for `Female`. A lower TPR means the model *misses more* true readmissions in that group.
- **Selection-rate gap:** 0.035 — groups are flagged for outreach at different rates.
- **AUROC gap:** 0.013 across reliable groups.

## By age

| group | count | selection_rate | tpr | fpr | precision | auroc | mean_pred | observed_rate |
|---|---|---|---|---|---|---|---|---|
| [0-10) | 29 | 0.034 | 0.000 | 0.034 | 0.000 | n/a | 0.037 | 0.000 |  ⚠️
| [10-20) | 109 | 0.092 | 0.400 | 0.077 | 0.200 | 0.790 | 0.055 | 0.046 |
| [20-30) | 204 | 0.157 | 0.667 | 0.116 | 0.312 | 0.780 | 0.074 | 0.074 |
| [30-40) | 537 | 0.130 | 0.314 | 0.118 | 0.157 | 0.708 | 0.068 | 0.065 |
| [40-50) | 1352 | 0.166 | 0.319 | 0.154 | 0.134 | 0.616 | 0.072 | 0.070 |
| [50-60) | 2466 | 0.142 | 0.354 | 0.124 | 0.197 | 0.664 | 0.069 | 0.079 |
| [60-70) | 3125 | 0.282 | 0.416 | 0.269 | 0.130 | 0.623 | 0.090 | 0.088 |
| [70-80) | 3606 | 0.376 | 0.611 | 0.349 | 0.170 | 0.662 | 0.101 | 0.105 |
| [80-90) | 2239 | 0.464 | 0.632 | 0.445 | 0.139 | 0.644 | 0.109 | 0.102 |
| [90-100) | 331 | 0.350 | 0.545 | 0.329 | 0.155 | 0.661 | 0.098 | 0.100 |

- **Recall (TPR) gap:** 0.352 — lowest for `[30-40)`, highest for `[20-30)`. A lower TPR means the model *misses more* true readmissions in that group.
- **Selection-rate gap:** 0.372 — groups are flagged for outreach at different rates.
- **AUROC gap:** 0.175 across reliable groups.

## Where this model is least reliable

Read the gaps above as a deployment caveat, not a verdict. The honest takeaways for a care team would be:

- Estimates for small subgroups (flagged ⚠️) are too noisy to act on — the first fairness finding is often *insufficient data*, which is itself a reason not to deploy blindly.
- Any subgroup with materially lower recall is one where outreach would systematically under-serve real readmissions; that group needs either a group-specific threshold or a human-in-the-loop safeguard before use.
- These results are on a historical (1999–2008) dataset and would need to be re-checked on local, contemporary data before any deployment.
