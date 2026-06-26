"""Sanity tests for the deterministic helpers in src/data_prep.py.

These cover the pure transforms (no dataset download required), so they run
fast in CI and guard the leakage-sensitive cleaning logic.
"""

import numpy as np
import pandas as pd

from src.data_prep import (
    TARGET,
    _age_band_to_midpoint,
    _map_icd9_to_group,
    build_feature_spec,
    clean,
)


def test_icd9_grouping():
    assert _map_icd9_to_group("250.83") == "Diabetes"
    assert _map_icd9_to_group("250") == "Diabetes"
    assert _map_icd9_to_group("410") == "Circulatory"
    assert _map_icd9_to_group("785") == "Circulatory"
    assert _map_icd9_to_group("486") == "Respiratory"
    assert _map_icd9_to_group("V45") == "Other"
    assert _map_icd9_to_group("E885") == "Other"
    assert _map_icd9_to_group(np.nan) == "Missing"


def test_age_band_midpoint():
    assert _age_band_to_midpoint("[0-10)") == 5.0
    assert _age_band_to_midpoint("[70-80)") == 75.0


def _toy_raw():
    """Minimal frame mirroring the raw schema clean() expects.

    Five encounters: patient 10 appears twice (dedupe), patient 20 expired
    (dropped), patient 30 has an invalid gender (dropped). Patients 10 and 40
    survive, so feature columns retain variance.
    """
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4, 5],
            "patient_nbr": [10, 10, 20, 30, 40],  # patient 10 appears twice
            "race": ["Caucasian", "Caucasian", "?", "Asian", "AfricanAmerican"],
            "gender": ["Male", "Male", "Female", "Unknown/Invalid", "Female"],
            "age": ["[60-70)", "[60-70)", "[40-50)", "[80-90)", "[50-60)"],
            "weight": ["?", "?", "?", "?", "?"],
            "payer_code": ["MC", "?", "?", "SP", "MC"],
            "medical_specialty": ["?", "Cardiology", "?", "?", "Surgery"],
            "discharge_disposition_id": [1, 1, 11, 1, 3],  # 11 = expired -> drop
            "admission_type_id": [1, 1, 2, 3, 2],
            "admission_source_id": [7, 7, 1, 4, 4],
            "diag_1": ["250.8", "410", "486", "715", "715"],
            "diag_2": ["401", "276", "?", "250", "276"],
            "diag_3": ["272", "?", "428", "414", "428"],
            "examide": ["No", "No", "No", "No", "No"],  # constant -> dropped
            "readmitted": ["<30", ">30", "NO", "<30", "NO"],
        }
    )


def test_clean_dedupe_target_and_drops():
    out = clean(_toy_raw())

    # Patient 10's duplicate encounter, the expired (patient 20) and the
    # invalid-gender (patient 30) rows are removed -> patients 10 and 40 remain.
    assert len(out) == 2
    row = out[out["age_midpoint"] == 65.0].iloc[0]  # patient 10's encounter

    # Binary target derived correctly (encounter 1 was "<30").
    assert TARGET in out.columns
    assert row[TARGET] == 1

    # Dropped columns and leakage-prone identifiers are gone.
    for col in ["weight", "payer_code", "encounter_id", "patient_nbr",
                "readmitted", "diag_1", "examide"]:
        assert col not in out.columns

    # Engineered features exist.
    assert row["diag_1_group"] == "Diabetes"
    assert row["age_midpoint"] == 65.0

    # No "?" sentinels survive and categoricals have no NaN.
    assert not (out.astype(str) == "?").any().any()
    numeric = out.select_dtypes(include=["number"]).columns
    categorical = out.columns.difference(numeric)
    assert not out[categorical].isna().any().any()


def test_feature_spec_roles():
    spec = build_feature_spec(clean(_toy_raw()))
    assert spec["target"] == TARGET
    # ID codes must be treated as categorical, not numeric.
    for idcol in ["admission_type_id", "discharge_disposition_id",
                  "admission_source_id"]:
        assert idcol in spec["categorical_features"]
    assert "age_midpoint" in spec["numeric_features"]
