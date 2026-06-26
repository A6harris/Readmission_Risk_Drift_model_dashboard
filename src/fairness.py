"""
fairness.py — Phase 4 of the readmission-monitoring project.

A single headline metric hides *who* a model fails. External-validation studies
repeatedly show clinical models performing far worse than advertised, with wide
variation across populations and sites [Wong et al. 2021; Lyons et al. 2023].
This module applies that local-validation principle to demographic subgroups:
it slices the held-out test set by ``race``, ``gender``, and ``age`` band and
reports, per subgroup, how the model actually behaves.

For each sensitive attribute we compute, per subgroup:

* **count** — how much evidence we have (small groups → unreliable estimates).
* **selection rate** — fraction flagged for outreach at the operating
  threshold (the disparity that matters if this gates a scarce resource).
* **TPR / FPR** — does the model catch readmissions, and how often does it
  false-alarm, equally across groups? (equalized-odds view)
* **precision (PPV)** — of those flagged, how many were truly readmitted.
* **AUROC** — per-group discrimination.
* **calibration** — mean predicted risk vs. observed rate within the group.

It then summarizes disparities (max−min gaps and Fairlearn's
demographic-parity / equalized-odds differences) and writes a plain-language
"where this model is least reliable" section.

Outputs:
  reports/figures/fairness_<attribute>.png
  reports/fairness.json
  reports/fairness.md
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
from fairlearn.metrics import (  # noqa: E402
    MetricFrame,
    count,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import precision_score, roc_auc_score  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# Operating threshold for the binary outreach decision. The decision-curve
# analysis (Phase 3) showed the model adds net benefit across ~0.04-0.58; we
# pick a low, illustrative operating point just above the base rate, where a
# capacity-constrained care team prioritizing outreach would plausibly sit.
DEFAULT_THRESHOLD = 0.10

# Subgroups smaller than this get their metrics reported but flagged as
# low-evidence (estimates are noisy and shouldn't drive conclusions).
MIN_RELIABLE_GROUP = 100


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
    return bundle, test_df, X, y, proba, spec


def _auroc_safe(y_true, y_score) -> float:
    """Per-group AUROC; NaN when a group has only one outcome class."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def subgroup_metrics(y_true, y_pred, proba, sensitive: pd.Series) -> pd.DataFrame:
    """Per-subgroup metric table for one sensitive attribute."""
    mf = MetricFrame(
        metrics={
            "count": count,
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
    table = mf.by_group.copy()

    # AUROC and calibration need scores, not the thresholded decision, so add
    # them per group by hand.
    df = pd.DataFrame({"y": y_true, "p": proba, "g": sensitive.to_numpy()})
    extra = df.groupby("g").apply(
        lambda d: pd.Series({
            "auroc": _auroc_safe(d["y"], d["p"]),
            "mean_pred": float(d["p"].mean()),
            "observed_rate": float(d["y"].mean()),
        }),
        include_groups=False,
    )
    table = table.join(extra)
    table["count"] = table["count"].astype(int)
    table["low_evidence"] = table["count"] < MIN_RELIABLE_GROUP
    return table


def _disparities(table: pd.DataFrame) -> dict:
    """Max-min gaps across *reliable* subgroups for the key metrics."""
    reliable = table[~table["low_evidence"]]
    out = {}
    for metric in ["selection_rate", "tpr", "fpr", "precision", "auroc"]:
        vals = reliable[metric].dropna()
        if len(vals) >= 2:
            out[f"{metric}_gap"] = float(vals.max() - vals.min())
            out[f"{metric}_min_group"] = str(vals.idxmin())
            out[f"{metric}_max_group"] = str(vals.idxmax())
        else:
            out[f"{metric}_gap"] = None
    return out


def plot_attribute(attr: str, table: pd.DataFrame, out_path: Path):
    """Grouped bar chart of the key per-subgroup metrics."""
    metrics = ["selection_rate", "tpr", "fpr", "auroc"]
    groups = table.index.astype(str).tolist()
    x = np.arange(len(groups))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.1), 5))
    for i, metric in enumerate(metrics):
        vals = table[metric].to_numpy(dtype=float)
        ax.bar(x + (i - 1.5) * width, np.nan_to_num(vals), width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("metric value")
    ax.set_title(f"Subgroup performance by {attr}", pad=34)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    # Mark low-evidence groups so the chart isn't over-read.
    for xi, g in zip(x, groups):
        if bool(table.loc[table.index.astype(str) == g, "low_evidence"].iloc[0]):
            ax.text(xi, 0.02, "n<100", ha="center", fontsize=8, color="grey")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_report(md_path: Path, threshold: float, results: dict):
    lines = [
        "# Phase 4 — Fairness Audit",
        "",
        f"Per-subgroup performance on the held-out test set, at an outreach "
        f"operating threshold of **{threshold:.2f}** (risk ≥ threshold → flag).",
        "",
        "> Fairness here means *reliability across groups*, not a single "
        "pass/fail number. Subgroups with n < 100 are reported but flagged as "
        "low-evidence and excluded from disparity gaps.",
        "",
    ]
    for attr, payload in results.items():
        table = payload["table"]
        disp = payload["disparities"]
        lines.append(f"## By {attr}")
        lines.append("")
        # Markdown table of rounded metrics.
        cols = ["count", "selection_rate", "tpr", "fpr", "precision",
                "auroc", "mean_pred", "observed_rate"]
        header = "| group | " + " | ".join(cols) + " |"
        sep = "|" + "---|" * (len(cols) + 1)
        lines += [header, sep]
        for grp, row in table.iterrows():
            cells = [str(grp)]
            for c in cols:
                v = row[c]
                if c == "count":
                    cells.append(str(int(v)))
                else:
                    cells.append("n/a" if pd.isna(v) else f"{v:.3f}")
            flag = "  ⚠️" if row["low_evidence"] else ""
            lines.append("| " + " | ".join(cells) + " |" + flag)
        lines.append("")
        # Narrative on the widest gaps.
        if disp.get("tpr_gap") is not None:
            lines.append(
                f"- **Recall (TPR) gap:** {disp['tpr_gap']:.3f} — lowest for "
                f"`{disp['tpr_min_group']}`, highest for `{disp['tpr_max_group']}`. "
                f"A lower TPR means the model *misses more* true readmissions in "
                f"that group."
            )
        if disp.get("selection_rate_gap") is not None:
            lines.append(
                f"- **Selection-rate gap:** {disp['selection_rate_gap']:.3f} — "
                f"groups are flagged for outreach at different rates."
            )
        if disp.get("auroc_gap") is not None:
            lines.append(
                f"- **AUROC gap:** {disp['auroc_gap']:.3f} across reliable groups."
            )
        lines.append("")

    lines += [
        "## Where this model is least reliable",
        "",
        "Read the gaps above as a deployment caveat, not a verdict. The honest "
        "takeaways for a care team would be:",
        "",
        "- Estimates for small subgroups (flagged ⚠️) are too noisy to act on — "
        "the first fairness finding is often *insufficient data*, which is "
        "itself a reason not to deploy blindly.",
        "- Any subgroup with materially lower recall is one where outreach would "
        "systematically under-serve real readmissions; that group needs either a "
        "group-specific threshold or a human-in-the-loop safeguard before use.",
        "- These results are on a historical (1999–2008) dataset and would need "
        "to be re-checked on local, contemporary data before any deployment.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    bundle, test_df, X, y, proba, spec = load_model_and_test(
        args.models_dir, args.processed_dir
    )
    y_pred = (proba >= args.threshold).astype(int)

    figures_dir = args.reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sensitive_attrs = spec.get("sensitive_features", ["race", "gender", "age"])
    results = {}
    json_payload = {"threshold": args.threshold, "attributes": {}}

    for attr in sensitive_attrs:
        table = subgroup_metrics(y, y_pred, proba, test_df[attr])
        disp = _disparities(table)
        plot_attribute(attr, table, figures_dir / f"fairness_{attr}.png")
        results[attr] = {"table": table, "disparities": disp}
        json_payload["attributes"][attr] = {
            "by_group": json.loads(
                table.reset_index().rename(columns={"index": "group"}).to_json(
                    orient="records"
                )
            ),
            "disparities": disp,
        }
        print(f"[fairness] {attr}: {len(table)} groups | "
              f"TPR gap={disp.get('tpr_gap')}")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "fairness.json").write_text(
        json.dumps(json_payload, indent=2), encoding="utf-8"
    )
    write_report(args.reports_dir / "fairness.md", args.threshold, results)

    print(f"[write] figures -> {figures_dir}")
    print(f"[write] {args.reports_dir / 'fairness.json'}")
    print(f"[write] {args.reports_dir / 'fairness.md'}")
    print("[done] Phase 4 complete.")


if __name__ == "__main__":
    main()
