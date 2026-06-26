"""
data_prep.py — Phase 1 of the readmission-monitoring project.

Downloads the UCI "Diabetes 130-US Hospitals for Years 1999-2008" dataset and
turns it into a clean, leakage-safe, stratified train/test split ready for
modeling.

What this script does (and why):

1.  Download + cache the raw dataset to ``data/raw/`` (idempotent).
2.  Decode the dataset's ``?`` sentinel as a real missing value.
3.  Drop columns that are mostly empty or non-predictive
    (``weight``, ``payer_code``) and drop zero-variance medication columns.
4.  Remove encounters that *cannot* be readmitted — patients discharged to
    hospice or who expired — because leaving them in leaks the outcome.
5.  De-duplicate to one encounter per patient (the first), following Strack
    et al. (2014). Multiple encounters from the same patient would otherwise
    leak information across the train/test boundary.
6.  Collapse the 3-class ``readmitted`` label to a binary
    ``readmitted_lt30`` target (1 = readmitted within 30 days).
7.  Group the ICD-9 ``diag_1/2/3`` codes into clinical categories.
8.  Engineer a numeric ``age_midpoint`` from the ``[x-y)`` age bands while
    keeping the band itself for the fairness audit.
9.  Write a stratified train/test split to ``data/processed/`` plus a
    ``feature_spec.json`` describing column roles.

DESIGN NOTE — encoding is intentionally deferred. We do *not* one-hot encode
or scale here. Those transforms must be *fit on the training split only* and
serialized together with the model (Phase 2) to avoid train/test leakage.
This script therefore emits clean, typed feature frames plus a feature spec;
``train.py`` builds the ``ColumnTransformer`` from that spec. This mirrors how
a real pipeline keeps preprocessing bound to the model it was fit with.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# UCI dataset id 296. The static export is a zip containing a
# ``dataset_diabetes/`` folder with ``diabetic_data.csv`` and ``IDS_mapping.csv``.
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
RAW_CSV_NAME = "diabetic_data.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Integer-coded columns that are really nominal categories, not quantities.
# They map to lookup tables in IDS_mapping.csv, so they must be one-hot encoded
# rather than treated as ordered numbers.
ID_CATEGORICAL_COLUMNS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

# Discharge dispositions that mean the patient died or went to hospice — these
# encounters can never produce a 30-day readmission, so keeping them leaks the
# outcome. (IDs per the dataset's IDS_mapping.csv.)
EXPIRED_OR_HOSPICE_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}

# Columns dropped outright: identifiers, mostly-missing, or non-predictive.
DROP_COLUMNS = ["weight", "payer_code", "encounter_id"]

# The 23 medication columns share the value set {No, Steady, Up, Down}.
MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Protected / sensitive attributes preserved for the Phase 4 fairness audit.
SENSITIVE_COLUMNS = ["race", "gender", "age"]

TARGET = "readmitted_lt30"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download_raw(raw_dir: Path) -> Path:
    """Download + extract the UCI dataset to ``raw_dir``; cache if present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / RAW_CSV_NAME
    if csv_path.exists():
        print(f"[download] cached raw dataset found at {csv_path}")
        return csv_path

    print(f"[download] fetching {UCI_ZIP_URL}")
    resp = requests.get(UCI_ZIP_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # Extract every member, flattening the dataset_diabetes/ prefix.
        for member in zf.namelist():
            name = Path(member).name
            if not name:  # directory entry
                continue
            with zf.open(member) as src:
                (raw_dir / name).write_bytes(src.read())

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected {RAW_CSV_NAME} in the UCI archive but it was not found."
        )
    print(f"[download] extracted raw dataset to {csv_path}")
    return csv_path


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #

def _map_icd9_to_group(code: object) -> str:
    """Map a single ICD-9 diagnosis code to a clinical category.

    Grouping follows Strack et al. (2014, Table 2): circulatory, respiratory,
    digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms,
    and an "other" bucket (which absorbs E/V codes and everything unmapped).
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    code = str(code)
    # E and V codes are supplementary classifications -> "Other".
    if code.startswith(("E", "V")):
        return "Other"
    try:
        num = float(code)
    except ValueError:
        return "Other"

    # Diabetes (250.xx) is called out specifically in the source paper.
    if int(num) == 250:
        return "Diabetes"
    if 390 <= num <= 459 or int(num) == 785:
        return "Circulatory"
    if 460 <= num <= 519 or int(num) == 786:
        return "Respiratory"
    if 520 <= num <= 579 or int(num) == 787:
        return "Digestive"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 580 <= num <= 629 or int(num) == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


def _age_band_to_midpoint(band: str) -> float:
    """Convert an ``[x-y)`` age band into its integer midpoint."""
    lo, hi = band.strip("[]()").split("-")
    return (int(lo) + int(hi)) / 2.0


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all deterministic, leakage-free cleaning steps."""
    n0 = len(df)

    # 1. The dataset uses "?" as a missing-value sentinel.
    df = df.replace("?", np.nan)

    # 2. Remove encounters that can't be readmitted (death / hospice).
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE_DISCHARGE_IDS)]
    print(f"[clean] removed {n0 - len(df)} expired/hospice encounters")

    # 3. Drop rows with an unusable gender label (a handful of records).
    df = df[df["gender"].isin(["Male", "Female"])]

    # 4. De-duplicate to the first encounter per patient (prevents leakage of
    #    a patient appearing in both train and test).
    n_before = len(df)
    df = df.sort_values("encounter_id").drop_duplicates(
        subset="patient_nbr", keep="first"
    )
    print(f"[clean] de-duplicated {n_before - len(df)} repeat-patient encounters")
    df = df.drop(columns=["patient_nbr"])

    # 5. Drop mostly-missing / non-predictive columns.
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # 6. Binary target: 1 if readmitted within 30 days, else 0.
    df[TARGET] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # 7. Group ICD-9 diagnoses into clinical categories.
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_group"] = df[col].map(_map_icd9_to_group)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # 8. Numeric age midpoint (keep the band for the fairness audit).
    df["age_midpoint"] = df["age"].map(_age_band_to_midpoint)

    # 9. Fill remaining categorical NaNs with an explicit "Missing" level so
    #    they survive encoding and stay auditable. This is deliberate for the
    #    lab columns (max_glu_serum, A1Cresult), where a missing value means the
    #    test was not ordered — itself a clinically informative signal. (Note:
    #    pandas parses their literal "None" string as NaN on read.)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.columns.difference(numeric_cols)
    df[categorical_cols] = df[categorical_cols].fillna("Missing")

    # 10. Cast nominal ID codes to strings so they encode as categories.
    for col in ID_CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # 11. Drop zero-variance feature columns (e.g. examide/citoglipton are
    #     always "No"). The target is always protected from this step.
    feature_cols = df.columns.drop(TARGET)
    nunique = df[feature_cols].nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols)
        print(f"[clean] dropped {len(constant_cols)} constant columns: "
              f"{constant_cols}")

    print(f"[clean] {n0} -> {len(df)} rows, {df.shape[1]} columns")
    return df.reset_index(drop=True)


def build_feature_spec(df: pd.DataFrame) -> dict:
    """Describe column roles so train.py can build its ColumnTransformer."""
    feature_df = df.drop(columns=[TARGET])
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]
    return {
        "target": TARGET,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "sensitive_features": [c for c in SENSITIVE_COLUMNS if c in df.columns],
        "n_rows": int(len(df)),
        "positive_rate": float(df[TARGET].mean()),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Imported here so the module's cleaning helpers can be unit-tested without
    # requiring scikit-learn to be installed.
    from sklearn.model_selection import train_test_split

    csv_path = download_raw(args.raw_dir)
    raw = pd.read_csv(csv_path, dtype={"diag_1": str, "diag_2": str, "diag_3": str})
    print(f"[load] raw shape: {raw.shape}")

    df = clean(raw)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df[TARGET],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.csv"
    test_path = args.out_dir / "test.csv"
    spec_path = args.out_dir / "feature_spec.json"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    with open(spec_path, "w") as f:
        json.dump(build_feature_spec(df), f, indent=2)

    print(
        f"[split] train={len(train_df)} (pos={train_df[TARGET].mean():.4f}) "
        f"test={len(test_df)} (pos={test_df[TARGET].mean():.4f})"
    )
    print(f"[write] {train_path}")
    print(f"[write] {test_path}")
    print(f"[write] {spec_path}")
    print("[done] Phase 1 complete.")


if __name__ == "__main__":
    main()
