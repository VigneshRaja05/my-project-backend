"""
====================================================================
PHASE 2D - FINAL EARLY WARNING DATASET CREATION
====================================================================

Project:
Explainable MSME-Oriented Loan Risk Prediction

Purpose:
Create final leakage-safe datasets for the Early Warning Model.

Two observation windows:

    Model B-3M
        First 3 months of payment behaviour
        -> Predict default AFTER month 3

    Model B-6M
        First 6 months of payment behaviour
        -> Predict default AFTER month 6

Target definition:

    0 = Loan did not default during the observed lifetime

    1 = Loan defaulted AFTER the observation window

Loans that defaulted WITHIN the observation window are excluded
because sufficient pre-default observation time does not exist.

IMPORTANT:
- No ML model is trained.
- No SMOTE is applied.
- No scaling is performed.
- No future payment information is used as a predictor.
- months_to_default is an audit variable only.
- default_date is NOT included as a predictor.
====================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ==================================================================
# 1. PROJECT PATHS
# ==================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "phase2" / "reports"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================================
# 2. INPUT FILES
# ==================================================================

LOAN_FILE = (
    RAW_DIR / "loan_portfolio.csv"
)

MODEL_B_3M_FILE = (
    PROCESSED_DIR
    / "model_b_early_warning_3m.parquet"
)

MODEL_B_6M_FILE = (
    PROCESSED_DIR
    / "model_b_early_warning_6m.parquet"
)


# ==================================================================
# 3. OUTPUT FILES
# ==================================================================

FINAL_3M_FILE = (
    PROCESSED_DIR
    / "model_b_final_early_warning_3m.parquet"
)

FINAL_6M_FILE = (
    PROCESSED_DIR
    / "model_b_final_early_warning_6m.parquet"
)


# ==================================================================
# 4. HELPER
# ==================================================================

def print_section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ==================================================================
# 5. LOAD ORIGINAL LOAN PORTFOLIO
# ==================================================================

print_section("LOADING ORIGINAL LOAN PORTFOLIO")

if not LOAN_FILE.exists():

    raise FileNotFoundError(
        f"Loan portfolio not found:\n{LOAN_FILE}"
    )

loan_df = pd.read_csv(
    LOAN_FILE,
    low_memory=False
)

print(
    f"Loan rows: {len(loan_df):,}"
)


# ==================================================================
# 6. VERIFY REQUIRED LOAN COLUMNS
# ==================================================================

print_section("VERIFYING LOAN COLUMNS")

required_loan_columns = [
    "loan_id",
    "defaulted",
    "origination_date",
    "default_date",
]

missing_columns = [
    col
    for col in required_loan_columns
    if col not in loan_df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required loan columns:\n"
        + "\n".join(missing_columns)
    )

print("All required loan columns are present.")


# ==================================================================
# 7. DATE CONVERSION
# ==================================================================

print_section("CONVERTING LOAN DATES")

loan_df["origination_date"] = pd.to_datetime(
    loan_df["origination_date"],
    errors="coerce"
)

loan_df["default_date"] = pd.to_datetime(
    loan_df["default_date"],
    errors="coerce"
)

loan_df["loan_id"] = (
    loan_df["loan_id"]
    .astype("string")
    .str.strip()
)


# ==================================================================
# 8. CHECK DATE QUALITY
# ==================================================================

print_section("DATE QUALITY CHECK")

missing_origination = (
    loan_df["origination_date"]
    .isna()
    .sum()
)

missing_default_date = (
    loan_df["default_date"]
    .isna()
    .sum()
)

print(
    f"Missing origination dates: "
    f"{missing_origination:,}"
)

print(
    f"Missing default dates: "
    f"{missing_default_date:,}"
)


# ==================================================================
# 9. VALIDATE DEFAULT DATE LOGIC
# ==================================================================

print_section("VALIDATING DEFAULT DATE LOGIC")

defaulted_df = loan_df[
    loan_df["defaulted"] == 1
].copy()

non_defaulted_df = loan_df[
    loan_df["defaulted"] == 0
].copy()

print(
    f"Defaulted loans: "
    f"{len(defaulted_df):,}"
)

print(
    f"Non-defaulted loans: "
    f"{len(non_defaulted_df):,}"
)

missing_default_for_defaulted = (
    defaulted_df["default_date"]
    .isna()
    .sum()
)

if missing_default_for_defaulted > 0:

    raise ValueError(
        "Some defaulted loans do not have a default_date."
    )


# ==================================================================
# 10. CALCULATE MONTHS TO DEFAULT
# ==================================================================

print_section("CALCULATING MONTHS TO DEFAULT")

loan_df["days_to_default"] = (
    loan_df["default_date"]
    - loan_df["origination_date"]
).dt.days

loan_df["months_to_default"] = (
    loan_df["days_to_default"]
    / 30.4375
)

# Non-defaulted loans have no default event.
loan_df.loc[
    loan_df["defaulted"] == 0,
    "months_to_default"
] = np.nan


# ==================================================================
# 11. CHECK INVALID DEFAULT TIMING
# ==================================================================

invalid_default_timing = (
    (loan_df["defaulted"] == 1)
    &
    (
        loan_df["months_to_default"] <= 0
    )
)

invalid_count = invalid_default_timing.sum()

print(
    f"Invalid default timing records: "
    f"{invalid_count:,}"
)

if invalid_count > 0:

    raise ValueError(
        "Default date occurs on/before origination date "
        "for one or more defaulted loans."
    )


# ==================================================================
# 12. LOAD MODEL B DATASETS
# ==================================================================

print_section("LOADING PHASE 2B DATASETS")

if not MODEL_B_3M_FILE.exists():

    raise FileNotFoundError(
        f"3-month dataset not found:\n"
        f"{MODEL_B_3M_FILE}"
    )

if not MODEL_B_6M_FILE.exists():

    raise FileNotFoundError(
        f"6-month dataset not found:\n"
        f"{MODEL_B_6M_FILE}"
    )


model_b_3m = pd.read_parquet(
    MODEL_B_3M_FILE
)

model_b_6m = pd.read_parquet(
    MODEL_B_6M_FILE
)

print(
    f"3-month rows: "
    f"{len(model_b_3m):,}"
)

print(
    f"6-month rows: "
    f"{len(model_b_6m):,}"
)


# ==================================================================
# 13. VERIFY LOAN IDs
# ==================================================================

print_section("VERIFYING LOAN ID INTEGRITY")

if "loan_id" not in model_b_3m.columns:

    raise ValueError(
        "loan_id missing from 3-month dataset."
    )

if "loan_id" not in model_b_6m.columns:

    raise ValueError(
        "loan_id missing from 6-month dataset."
    )

model_b_3m["loan_id"] = (
    model_b_3m["loan_id"]
    .astype("string")
    .str.strip()
)

model_b_6m["loan_id"] = (
    model_b_6m["loan_id"]
    .astype("string")
    .str.strip()
)

print("Loan IDs validated.")


# ==================================================================
# 14. PREPARE LOAN TIMING INFORMATION
# ==================================================================

timing_columns = [
    "loan_id",
    "defaulted",
    "origination_date",
    "default_date",
    "months_to_default",
]

timing_df = loan_df[
    timing_columns
].copy()


# ==================================================================
# 15. MERGE TIMING WITH 3M DATA
# ==================================================================

print_section("PREPARING 3-MONTH EARLY WARNING DATA")

final_3m = model_b_3m.merge(
    timing_df,
    on="loan_id",
    how="left",
    validate="one_to_one",
    suffixes=("", "_portfolio")
)

print(
    f"Rows after timing merge: "
    f"{len(final_3m):,}"
)


# ==================================================================
# 16. VERIFY MERGE
# ==================================================================

missing_target_3m = (
    final_3m["defaulted"]
    .isna()
    .sum()
)

if missing_target_3m > 0:

    raise ValueError(
        f"{missing_target_3m:,} "
        "3-month records could not be matched "
        "to loan portfolio."
    )


# ==================================================================
# 17. CREATE 3M TARGET
# ==================================================================

OBSERVATION_3M = 3

final_3m["excluded_early_default_3m"] = (
    (final_3m["defaulted"] == 1)
    &
    (
        final_3m["months_to_default"]
        <= OBSERVATION_3M
    )
)

final_3m["early_warning_target_3m"] = np.where(
    final_3m["defaulted"] == 1,
    np.where(
        final_3m["months_to_default"]
        > OBSERVATION_3M,
        1,
        np.nan
    ),
    0
)


# ==================================================================
# 18. REMOVE EARLY 3M DEFAULTS
# ==================================================================

early_defaults_3m = (
    final_3m[
        final_3m["excluded_early_default_3m"]
    ]
    .copy()
)

print(
    f"Loans defaulting within 3 months: "
    f"{len(early_defaults_3m):,}"
)


final_3m = final_3m[
    ~final_3m["excluded_early_default_3m"]
].copy()


# ==================================================================
# 19. VALIDATE 3M TARGET
# ==================================================================

if final_3m["early_warning_target_3m"].isna().any():

    raise ValueError(
        "Unexpected missing 3-month target."
    )

final_3m["early_warning_target_3m"] = (
    final_3m["early_warning_target_3m"]
    .astype(int)
)


# ==================================================================
# 20. 3M TARGET VALIDATION
# ==================================================================

print_section("3-MONTH TARGET VALIDATION")

print(
    final_3m[
        "early_warning_target_3m"
    ]
    .value_counts()
    .sort_index()
)

invalid_3m_targets = (
    (
        final_3m[
            "early_warning_target_3m"
        ] == 1
    )
    &
    (
        final_3m[
            "months_to_default"
        ] <= OBSERVATION_3M
    )
).sum()

print(
    f"Invalid target records: "
    f"{invalid_3m_targets:,}"
)

if invalid_3m_targets > 0:

    raise ValueError(
        "3-month target leakage detected."
    )


# ==================================================================
# 21. MERGE TIMING WITH 6M DATA
# ==================================================================

print_section("PREPARING 6-MONTH EARLY WARNING DATA")

final_6m = model_b_6m.merge(
    timing_df,
    on="loan_id",
    how="left",
    validate="one_to_one",
    suffixes=("", "_portfolio")
)

print(
    f"Rows after timing merge: "
    f"{len(final_6m):,}"
)


# ==================================================================
# 22. VERIFY 6M MERGE
# ==================================================================

missing_target_6m = (
    final_6m["defaulted"]
    .isna()
    .sum()
)

if missing_target_6m > 0:

    raise ValueError(
        f"{missing_target_6m:,} "
        "6-month records could not be matched "
        "to loan portfolio."
    )


# ==================================================================
# 23. CREATE 6M TARGET
# ==================================================================

OBSERVATION_6M = 6

final_6m["excluded_early_default_6m"] = (
    (final_6m["defaulted"] == 1)
    &
    (
        final_6m["months_to_default"]
        <= OBSERVATION_6M
    )
)

final_6m["early_warning_target_6m"] = np.where(
    final_6m["defaulted"] == 1,
    np.where(
        final_6m["months_to_default"]
        > OBSERVATION_6M,
        1,
        np.nan
    ),
    0
)


# ==================================================================
# 24. REMOVE EARLY 6M DEFAULTS
# ==================================================================

early_defaults_6m = (
    final_6m[
        final_6m["excluded_early_default_6m"]
    ]
    .copy()
)

print(
    f"Loans defaulting within 6 months: "
    f"{len(early_defaults_6m):,}"
)


final_6m = final_6m[
    ~final_6m["excluded_early_default_6m"]
].copy()


# ==================================================================
# 25. VALIDATE 6M TARGET
# ==================================================================

if final_6m["early_warning_target_6m"].isna().any():

    raise ValueError(
        "Unexpected missing 6-month target."
    )

final_6m["early_warning_target_6m"] = (
    final_6m["early_warning_target_6m"]
    .astype(int)
)


# ==================================================================
# 26. 6M TARGET VALIDATION
# ==================================================================

print_section("6-MONTH TARGET VALIDATION")

print(
    final_6m[
        "early_warning_target_6m"
    ]
    .value_counts()
    .sort_index()
)

invalid_6m_targets = (
    (
        final_6m[
            "early_warning_target_6m"
        ] == 1
    )
    &
    (
        final_6m[
            "months_to_default"
        ] <= OBSERVATION_6M
    )
).sum()

print(
    f"Invalid target records: "
    f"{invalid_6m_targets:,}"
)

if invalid_6m_targets > 0:

    raise ValueError(
        "6-month target leakage detected."
    )


# ==================================================================
# 27. REMOVE LEAKAGE / AUDIT-ONLY COLUMNS
# ==================================================================

print_section(
    "REMOVING TARGET-LEAKAGE AND AUDIT-ONLY COLUMNS"
)

# These columns are useful for auditing but must NOT become
# predictors during Phase 3.

AUDIT_ONLY_COLUMNS = [
    "defaulted",
    "origination_date",
    "default_date",
    "days_to_default",
    "months_to_default",
    "excluded_early_default_3m",
    "excluded_early_default_6m",
]


def remove_audit_columns(df, target):

    columns_to_remove = [
        col
        for col in AUDIT_ONLY_COLUMNS
        if col in df.columns
        and col != target
    ]

    return df.drop(
        columns=columns_to_remove
    )


final_3m_model = remove_audit_columns(
    final_3m,
    "early_warning_target_3m"
)

final_6m_model = remove_audit_columns(
    final_6m,
    "early_warning_target_6m"
)


# ==================================================================
# 28. VERIFY NO FORBIDDEN COLUMNS
# ==================================================================

print_section(
    "FINAL LEAKAGE COLUMN CHECK"
)

for name, dataset, target in [
    (
        "3M",
        final_3m_model,
        "early_warning_target_3m"
    ),
    (
        "6M",
        final_6m_model,
        "early_warning_target_6m"
    ),
]:

    forbidden_columns = [
        col
        for col in [
            "default_date",
            "months_to_default",
            "days_to_default",
            "defaulted",
        ]
        if col in dataset.columns
    ]

    print(
        f"{name} forbidden columns: "
        f"{forbidden_columns}"
    )

    if forbidden_columns:

        raise ValueError(
            f"{name} dataset contains "
            "forbidden leakage columns."
        )


# ==================================================================
# 29. CHECK DUPLICATE LOANS
# ==================================================================

print_section("FINAL DUPLICATE CHECK")

duplicate_3m = (
    final_3m_model["loan_id"]
    .duplicated()
    .sum()
)

duplicate_6m = (
    final_6m_model["loan_id"]
    .duplicated()
    .sum()
)

print(
    f"3M duplicate loan IDs: "
    f"{duplicate_3m:,}"
)

print(
    f"6M duplicate loan IDs: "
    f"{duplicate_6m:,}"
)

if duplicate_3m > 0 or duplicate_6m > 0:

    raise ValueError(
        "Duplicate loan IDs detected."
    )


# ==================================================================
# 30. TARGET DISTRIBUTION REPORT
# ==================================================================

print_section("FINAL TARGET DISTRIBUTIONS")

target_report = []

for name, dataset, target in [
    (
        "3M",
        final_3m_model,
        "early_warning_target_3m"
    ),
    (
        "6M",
        final_6m_model,
        "early_warning_target_6m"
    ),
]:

    counts = (
        dataset[target]
        .value_counts()
        .sort_index()
    )

    total = len(dataset)

    for value, count in counts.items():

        target_report.append({
            "dataset": name,
            "target": target,
            "class": int(value),
            "count": int(count),
            "percentage": (
                count / total * 100
            )
        })

target_report_df = pd.DataFrame(
    target_report
)

print(
    target_report_df.to_string(
        index=False
    )
)

target_report_df.to_csv(
    REPORT_DIR
    / "final_early_warning_target_distribution.csv",
    index=False
)


# ==================================================================
# 31. EXCLUSION REPORT
# ==================================================================

print_section("EARLY DEFAULT EXCLUSION REPORT")

exclusion_report = pd.DataFrame({
    "dataset": [
        "3M",
        "6M"
    ],
    "observation_window_months": [
        3,
        6
    ],
    "early_defaults_excluded": [
        len(early_defaults_3m),
        len(early_defaults_6m)
    ],
    "final_rows": [
        len(final_3m_model),
        len(final_6m_model)
    ]
})

exclusion_report[
    "original_rows"
] = [
    len(model_b_3m),
    len(model_b_6m)
]

exclusion_report[
    "retention_percentage"
] = (
    exclusion_report["final_rows"]
    / exclusion_report["original_rows"]
    * 100
)

print(
    exclusion_report.to_string(
        index=False
    )
)

exclusion_report.to_csv(
    REPORT_DIR
    / "early_default_exclusion_report.csv",
    index=False
)


# ==================================================================
# 32. FEATURE LIST
# ==================================================================

print_section("FINAL FEATURE LIST")

feature_records = []

for name, dataset, target in [
    (
        "3M",
        final_3m_model,
        "early_warning_target_3m"
    ),
    (
        "6M",
        final_6m_model,
        "early_warning_target_6m"
    ),
]:

    for col in dataset.columns:

        if col == "loan_id":

            role = "identifier"

        elif col == target:

            role = "target"

        else:

            role = "predictor"

        feature_records.append({
            "dataset": name,
            "feature": col,
            "role": role,
            "dtype": str(
                dataset[col].dtype
            )
        })

feature_report_df = pd.DataFrame(
    feature_records
)

feature_report_df.to_csv(
    REPORT_DIR
    / "final_early_warning_feature_list.csv",
    index=False
)

print(
    feature_report_df.to_string(
        index=False
    )
)


# ==================================================================
# 33. FINAL DATASET SHAPES
# ==================================================================

print_section("FINAL DATASET SHAPES")

print(
    f"3M final shape: "
    f"{final_3m_model.shape}"
)

print(
    f"6M final shape: "
    f"{final_6m_model.shape}"
)


# ==================================================================
# 34. SAVE FINAL DATASETS
# ==================================================================

print_section("SAVING FINAL DATASETS")

final_3m_model.to_parquet(
    FINAL_3M_FILE,
    index=False
)

final_6m_model.to_parquet(
    FINAL_6M_FILE,
    index=False
)

print(
    f"3M saved to:\n"
    f"{FINAL_3M_FILE}"
)

print(
    f"6M saved to:\n"
    f"{FINAL_6M_FILE}"
)


# ==================================================================
# 35. FINAL VERIFICATION
# ==================================================================

print_section("FINAL VERIFICATION")

for name, dataset, target in [
    (
        "3M",
        final_3m_model,
        "early_warning_target_3m"
    ),
    (
        "6M",
        final_6m_model,
        "early_warning_target_6m"
    ),
]:

    print(
        f"\n{name} DATASET"
    )

    print(
        f"Rows    : {len(dataset):,}"
    )

    print(
        f"Columns : {len(dataset.columns):,}"
    )

    print(
        f"Target  : {target}"
    )

    print(
        "Target distribution:"
    )

    print(
        dataset[target]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "Missing target:"
        f" {dataset[target].isna().sum():,}"
    )


# ==================================================================
# 36. COMPLETION
# ==================================================================

print_section(
    "PHASE 2D COMPLETE"
)

print(
    """
Final Early Warning datasets successfully created.

MODEL B-3M:
    First 3 months of behaviour
    Predict future default after month 3

MODEL B-6M:
    First 6 months of behaviour
    Predict future default after month 6

Early defaults within the corresponding observation window
were excluded.

Leakage-sensitive outcome columns were removed from the
final modelling datasets.

No ML model was trained.
No SMOTE was applied.
No scaling was applied.
"""
)