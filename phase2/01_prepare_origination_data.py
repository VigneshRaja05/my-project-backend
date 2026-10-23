"""
====================================================================
PHASE 2A - ORIGINATION DATA PREPARATION
====================================================================

Project:
Explainable MSME-Oriented Loan Risk Prediction

Model:
Model A - Origination Risk Model

Objective:
Create a clean, leakage-safe, modelling-ready dataset using
information available at loan origination.

IMPORTANT:
- Raw data is never modified.
- No ML model is trained.
- No SMOTE is performed.
- No scaling is performed.
- No test-set information is used for preprocessing decisions.
- Post-outcome variables are excluded.
====================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ==================================================================
# 1. PATHS
# ==================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "phase2" / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================================
# 2. FILE
# ==================================================================

INPUT_FILE = RAW_DIR / "loan_portfolio.csv"

OUTPUT_FILE = (
    PROCESSED_DIR / "model_a_origination_dataset.parquet"
)


# ==================================================================
# 3. TARGET
# ==================================================================

TARGET = "defaulted"


# ==================================================================
# 4. INITIAL FEATURE SET
# ==================================================================

CATEGORICAL_FEATURES = [
    "sector",
    "loan_type",
    "collateral",
    "initial_rating",
]

NUMERICAL_FEATURES = [
    "maturity_months",
    "credit_score",
    "ead",
    "coupon_rate",
    "leverage",
    "interest_coverage",
    "debt_to_equity",
]


# ==================================================================
# 5. EXCLUDED FEATURES
# ==================================================================

EXCLUDED_FEATURES = {
    "loan_id": "Identifier only",
    "default_date": "Post-outcome information / leakage",
    "survival_months": "Outcome-period information / leakage",
    "recovery_rate": "Post-default information",
    "loss_given_default": "Post-default information",
    "pd_annual": "Existing PD estimate; reserved for benchmark",
    "lgd": "Derived risk-management quantity",
    "el": "Derived expected-loss quantity",
    "unexpected_loss": "Derived risk quantity",
    "rwa": "Derived risk-weighted asset quantity",
}


# ==================================================================
# 6. HELPER
# ==================================================================

def print_section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ==================================================================
# 7. LOAD DATA
# ==================================================================

print_section("LOADING LOAN PORTFOLIO")

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ==================================================================
# 8. VERIFY REQUIRED COLUMNS
# ==================================================================

print_section("VERIFYING REQUIRED COLUMNS")

required_columns = (
    CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
    + [TARGET]
)

missing_required = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_required:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_required)
    )

print("All required columns are present.")


# ==================================================================
# 9. TARGET VALIDATION
# ==================================================================

print_section("TARGET VALIDATION")

print(
    df[TARGET]
    .value_counts(dropna=False)
    .sort_index()
)

if df[TARGET].isna().any():

    raise ValueError(
        "Target contains missing values."
    )

unique_targets = set(
    df[TARGET].unique()
)

if not unique_targets.issubset({0, 1}):

    raise ValueError(
        f"Unexpected target values: {unique_targets}"
    )

print("\nTarget validation passed.")


# ==================================================================
# 10. DUPLICATE LOAN CHECK
# ==================================================================

print_section("DUPLICATE LOAN CHECK")

if "loan_id" in df.columns:

    duplicate_ids = df["loan_id"].duplicated().sum()

    print(
        f"Duplicate loan IDs: {duplicate_ids:,}"
    )

    if duplicate_ids > 0:

        raise ValueError(
            "Duplicate loan IDs detected."
        )


# ==================================================================
# 11. CREATE CLEAN MODEL DATASET
# ==================================================================

print_section("CREATING MODEL A DATASET")

model_columns = (
    ["loan_id"]
    + CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
    + [TARGET]
)

model_df = df[model_columns].copy()

print(
    f"Initial Model A shape: "
    f"{model_df.shape}"
)


# ==================================================================
# 12. DATA TYPE CLEANING
# ==================================================================

print_section("DATA TYPE CLEANING")

for col in NUMERICAL_FEATURES:

    model_df[col] = pd.to_numeric(
        model_df[col],
        errors="coerce"
    )

for col in CATEGORICAL_FEATURES:

    model_df[col] = (
        model_df[col]
        .astype("string")
        .str.strip()
    )

print("Data types normalized.")


# ==================================================================
# 13. MISSING VALUE AUDIT
# ==================================================================

print_section("MISSING VALUE AUDIT")

missing_report = (
    model_df
    .isna()
    .sum()
    .reset_index()
)

missing_report.columns = [
    "feature",
    "missing_count"
]

missing_report["missing_percentage"] = (
    missing_report["missing_count"]
    / len(model_df)
    * 100
)

print(
    missing_report.to_string(index=False)
)

missing_report.to_csv(
    REPORT_DIR / "model_a_missing_values.csv",
    index=False
)


# ==================================================================
# 14. DO NOT IMPUTE YET
# ==================================================================

"""
We intentionally do NOT perform model imputation here.

Reason:

Imputation parameters should be learned from the training
partition only.

Phase 3 will build the sklearn preprocessing pipeline after
the train/validation/test split.

Therefore this dataset retains NaN values if they exist.
"""


# ==================================================================
# 15. NUMERICAL SANITY CHECK
# ==================================================================

print_section("NUMERICAL SANITY CHECK")

numeric_report = []

for col in NUMERICAL_FEATURES:

    series = model_df[col]

    numeric_report.append({
        "feature": col,
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "missing": series.isna().sum(),
        "negative_values": (
            (series < 0).sum()
        )
    })

numeric_report_df = pd.DataFrame(
    numeric_report
)

print(
    numeric_report_df.to_string(index=False)
)

numeric_report_df.to_csv(
    REPORT_DIR / "model_a_numeric_sanity.csv",
    index=False
)


# ==================================================================
# 16. CATEGORICAL SANITY CHECK
# ==================================================================

print_section("CATEGORICAL SANITY CHECK")

categorical_report = []

for col in CATEGORICAL_FEATURES:

    series = model_df[col]

    categorical_report.append({
        "feature": col,
        "unique_values": series.nunique(
            dropna=True
        ),
        "missing": series.isna().sum(),
        "top_value": series.mode(
            dropna=True
        ).iloc[0]
        if not series.dropna().empty
        else None
    })

categorical_report_df = pd.DataFrame(
    categorical_report
)

print(
    categorical_report_df.to_string(index=False)
)

categorical_report_df.to_csv(
    REPORT_DIR / "model_a_categorical_sanity.csv",
    index=False
)


# ==================================================================
# 17. MODEL A FEATURE LIST
# ==================================================================

print_section("MODEL A FEATURE LIST")

feature_list = []

for col in CATEGORICAL_FEATURES:

    feature_list.append({
        "feature": col,
        "type": "categorical",
        "role": "predictor"
    })

for col in NUMERICAL_FEATURES:

    feature_list.append({
        "feature": col,
        "type": "numerical",
        "role": "predictor"
    })

feature_list.append({
    "feature": TARGET,
    "type": "binary",
    "role": "target"
})

feature_list_df = pd.DataFrame(
    feature_list
)

feature_list_df.to_csv(
    REPORT_DIR / "model_a_feature_list.csv",
    index=False
)

print(
    feature_list_df.to_string(index=False)
)


# ==================================================================
# 18. EXCLUSION REPORT
# ==================================================================

print_section("EXCLUDED FEATURES")

exclusion_records = []

for feature, reason in EXCLUDED_FEATURES.items():

    exclusion_records.append({
        "feature": feature,
        "decision": "EXCLUDE",
        "reason": reason
    })

exclusion_df = pd.DataFrame(
    exclusion_records
)

print(
    exclusion_df.to_string(index=False)
)

exclusion_df.to_csv(
    REPORT_DIR / "model_a_excluded_features.csv",
    index=False
)


# ==================================================================
# 19. TARGET DISTRIBUTION
# ==================================================================

print_section("TARGET DISTRIBUTION")

target_distribution = (
    model_df[TARGET]
    .value_counts()
    .sort_index()
    .reset_index()
)

target_distribution.columns = [
    "defaulted",
    "count"
]

target_distribution["percentage"] = (
    target_distribution["count"]
    / len(model_df)
    * 100
)

print(
    target_distribution.to_string(index=False)
)

target_distribution.to_csv(
    REPORT_DIR / "model_a_target_distribution.csv",
    index=False
)


# ==================================================================
# 20. SAVE MODEL A DATASET
# ==================================================================

print_section("SAVING MODEL A DATASET")

model_df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print(
    f"Saved:\n{OUTPUT_FILE}"
)

print(
    f"Final shape: {model_df.shape}"
)


# ==================================================================
# 21. FINAL SUMMARY
# ==================================================================

print_section("PHASE 2A COMPLETE")

print(
    f"""
Model A:
    Purpose:
        Origination default-risk prediction

    Rows:
        {len(model_df):,}

    Predictors:
        {len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES)}

    Categorical:
        {len(CATEGORICAL_FEATURES)}

    Numerical:
        {len(NUMERICAL_FEATURES)}

    Target:
        {TARGET}

    Output:
        {OUTPUT_FILE}

No model was trained.
No SMOTE was applied.
No scaling was applied.
No raw data was modified.
"""
)