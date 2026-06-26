"""Fast tests for the pipeline construction in src/train.py.

These fit on a tiny synthetic frame so they run in milliseconds — they verify
the pipeline is wired correctly (preprocessing + estimator, leakage-safe,
handles unseen categories), not model quality.
"""

import numpy as np
import pandas as pd

from src.train import build_models, build_preprocessor, evaluate


def _toy_spec_and_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    spec = {
        "target": "y",
        "numeric_features": ["age_midpoint", "num_medications"],
        "categorical_features": ["race", "admission_type_id"],
    }
    df = pd.DataFrame(
        {
            "age_midpoint": rng.normal(60, 15, n),
            "num_medications": rng.integers(1, 30, n),
            "race": rng.choice(["Caucasian", "AfricanAmerican", "Asian"], n),
            "admission_type_id": rng.choice(["1", "2", "3"], n),
        }
    )
    # Signal so the models can do better than chance.
    y = ((df["age_midpoint"] > 60) | (df["race"] == "Asian")).astype(int)
    return spec, df, y.to_numpy()


def test_preprocessor_shapes():
    spec, df, _ = _toy_spec_and_data()
    pre = build_preprocessor(spec)
    out = pre.fit_transform(df)
    # 2 numeric + one-hot of (3 race + 3 admission_type) = 8 columns.
    assert out.shape[0] == len(df)
    assert out.shape[1] == 2 + 3 + 3


def test_models_fit_predict_and_handle_unknown_categories():
    spec, df, y = _toy_spec_and_data()
    models = build_models(spec, n_pos=int(y.sum()), n_neg=int((1 - y).sum()))
    assert set(models) == {"logistic_regression", "xgboost"}

    for name, pipe in models.items():
        pipe.fit(df, y)

        # A category level never seen in training must not crash prediction.
        unseen = df.iloc[:5].copy()
        unseen["race"] = "Martian"
        proba = pipe.predict_proba(unseen)[:, 1]
        assert proba.shape == (5,)
        assert np.all((proba >= 0) & (proba <= 1))

        # Discrimination metrics are well-formed on the training data.
        m = evaluate(pipe, df, y)
        assert 0.0 <= m["auroc"] <= 1.0
        assert 0.0 <= m["auprc"] <= 1.0
