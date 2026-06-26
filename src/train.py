"""
train.py — Phase 2 of the readmission-monitoring project.

Trains two models on the processed data from ``data_prep.py``:

1.  A regularized **logistic regression** — an interpretable, well-calibrated
    baseline. Every health-AI evaluation should have one; if a black box can't
    beat a transparent linear model, the complexity isn't earning its keep.
2.  An **XGBoost** gradient-boosted model — the main candidate.

Both are wrapped in the *same* scikit-learn ``Pipeline`` so that all
preprocessing (imputation, scaling, one-hot encoding) is **fit on the training
fold only** and travels with the model when serialized. This is the single most
important guard against train/serve skew and evaluation leakage.

Model selection uses stratified cross-validation on the training set (criterion:
average precision / AUPRC, which is the honest metric for a ~9% positive class).
The winner is refit on the full training set and serialized to
``models/model.joblib`` together with its feature spec; per-model metrics on the
held-out test set are written to ``models/metrics.json`` for the report phases.

We deliberately do not grid-search hard here — the project's value is in the
governance layer, not in squeezing out the last point of AUC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

SEED = 42


# --------------------------------------------------------------------------- #
# Data + pipeline construction
# --------------------------------------------------------------------------- #

def load_data(processed_dir: Path):
    """Load train/test frames and the feature spec written by data_prep.py."""
    with open(processed_dir / "feature_spec.json") as f:
        spec = json.load(f)
    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    # ID codes are written as strings by data_prep, but pandas re-infers ints on
    # read; cast the declared categoricals back to string so the encoder treats
    # them as nominal.
    for col in spec["categorical_features"]:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    return train_df, test_df, spec


def build_preprocessor(spec: dict) -> ColumnTransformer:
    """Scale numeric features and one-hot encode categoricals.

    ``handle_unknown="ignore"`` is essential: at monitoring time (Phase 6) the
    incoming data may contain category levels never seen in training, and the
    pipeline must not crash on them.
    """
    numeric = spec["numeric_features"]
    categorical = spec["categorical_features"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )


def build_models(spec: dict, n_pos: int, n_neg: int) -> dict[str, Pipeline]:
    """Return the candidate pipelines, each = preprocessor + estimator."""
    pre = build_preprocessor

    # Both models address the ~9% class imbalance: LR via class_weight, XGBoost
    # via scale_pos_weight (= negatives / positives).
    scale_pos_weight = n_neg / n_pos

    logreg = Pipeline([
        ("pre", pre(spec)),
        ("clf", LogisticRegression(
            # L2 is the default; C controls the regularization strength.
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=SEED,
        )),
    ])

    xgb = Pipeline([
        ("pre", pre(spec)),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED,
        )),
    ])

    return {"logistic_regression": logreg, "xgboost": xgb}


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #

def evaluate(pipe: Pipeline, X, y) -> dict:
    """Test-set discrimination metrics for a fitted pipeline."""
    proba = pipe.predict_proba(X)[:, 1]
    return {
        "auroc": float(roc_auc_score(y, proba)),
        "auprc": float(average_precision_score(y, proba)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    train_df, test_df, spec = load_data(args.processed_dir)
    target = spec["target"]

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target].to_numpy()
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target].to_numpy()

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    print(f"[train] {len(X_train)} rows | positives={n_pos} ({y_train.mean():.4f})")

    models = build_models(spec, n_pos=n_pos, n_neg=n_neg)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)

    # --- Model selection: cross-validated AUPRC on the training set only. -----
    results = {}
    for name, pipe in models.items():
        print(f"[cv] {name}: running {args.cv_folds}-fold CV ...")
        cv_auprc = cross_val_score(
            pipe, X_train, y_train, scoring="average_precision", cv=cv, n_jobs=-1
        )
        cv_auroc = cross_val_score(
            pipe, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1
        )
        results[name] = {
            "cv_auprc_mean": float(cv_auprc.mean()),
            "cv_auprc_std": float(cv_auprc.std()),
            "cv_auroc_mean": float(cv_auroc.mean()),
            "cv_auroc_std": float(cv_auroc.std()),
        }
        print(f"      AUPRC={cv_auprc.mean():.4f}+/-{cv_auprc.std():.4f}  "
              f"AUROC={cv_auroc.mean():.4f}+/-{cv_auroc.std():.4f}")

    best_name = max(results, key=lambda n: results[n]["cv_auprc_mean"])
    print(f"[select] best by CV AUPRC: {best_name}")

    # --- Refit every model on full train; record held-out test metrics. -------
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        results[name]["test"] = evaluate(pipe, X_test, y_test)
        print(f"[test] {name}: AUROC={results[name]['test']['auroc']:.4f}  "
              f"AUPRC={results[name]['test']['auprc']:.4f}")

    best_pipe = models[best_name]

    # --- Serialize the winning pipeline + metadata. ---------------------------
    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / "model.joblib"
    joblib.dump(
        {
            "pipeline": best_pipe,
            "model_name": best_name,
            "feature_spec": spec,
            "target": target,
        },
        model_path,
    )

    metrics = {
        "best_model": best_name,
        "selection_metric": "cv_auprc_mean",
        "test_positive_rate": float(y_test.mean()),
        "models": results,
    }
    metrics_path = args.models_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[write] {model_path}  (serialized: {best_name} + preprocessing)")
    print(f"[write] {metrics_path}")
    print("[done] Phase 2 complete.")


if __name__ == "__main__":
    main()
