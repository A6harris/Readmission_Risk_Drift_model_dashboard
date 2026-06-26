"""Test the encoded-column -> source-variable mapping in src/explain.py.

This is the bit that makes the aggregated SHAP importance trustworthy: if the
mapping is off by one, one-hot columns get attributed to the wrong variable.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.explain import original_feature_map


def test_original_feature_map_aligns_with_encoder():
    spec = {
        "numeric_features": ["age_midpoint", "num_medications"],
        "categorical_features": ["race", "gender"],
    }
    df = pd.DataFrame(
        {
            "age_midpoint": [50.0, 60.0, 70.0, 80.0],
            "num_medications": [1, 2, 3, 4],
            "race": ["A", "B", "A", "C"],   # 3 levels
            "gender": ["M", "F", "M", "F"],  # 2 levels
        }
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), spec["numeric_features"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             spec["categorical_features"]),
        ]
    )
    pre.fit(df)
    encoded = list(pre.get_feature_names_out())
    mapping = original_feature_map(spec, encoded, pre)

    # One entry per encoded column.
    assert len(mapping) == len(encoded)
    # 2 numeric + 3 race + 2 gender = 7 columns.
    assert len(mapping) == 7
    # Order: numerics first (1:1), then race x3, then gender x2.
    assert mapping[:2] == ["age_midpoint", "num_medications"]
    assert mapping[2:5] == ["race", "race", "race"]
    assert mapping[5:7] == ["gender", "gender"]

    # Each mapping entry should be consistent with the encoded column's prefix.
    for src, enc in zip(mapping, encoded):
        assert enc.split("__", 1)[-1].startswith(src)
