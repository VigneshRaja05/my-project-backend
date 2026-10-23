"""
====================================================================
PHASE 2B - PAYMENT BEHAVIOUR FEATURE ENGINEERING
====================================================================

Project:
Explainable MSME-Oriented Loan Risk Prediction

Model:
Model B - Early Warning Model

Objective:
Create leakage-aware behavioural features from payment history.

Observation windows:
    3 months
    6 months

IMPORTANT:
Only payment observations within the selected observation window
are used to construct behavioural features.

No ML model is trained here.
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
# 2. FILES
# ==================================================================

LOAN_FILE = RAW_DIR / "loan_portfolio.csv"
PAYMENT_FILE = RAW_DIR / "payment_history.csv"


# ==================================================================
# 3. OUTPUTS
# ==================================================================

OUTPUT_3M = (
    PROCESSED_DIR
    / "model_b_early_warning_3m.parquet"
)

OUTPUT_6M = (
    PROCESSED_DIR
    / "model_b_early_warning_6m.parquet"
)


# ==================================================================
# 4. REQUIRED PAYMENT COLUMNS
# ==================================================================

PAYMENT_COLUMNS = [
    "loan_id",
    "month_on_book",
    "payment_date",
    "scheduled_payment",
    "actual_payment",
    "outstanding_balance",
    "interest_component",
    "principal_component",
    "dpd",
    "dpd_bucket",
    "missed_payment",
    "cumulative_missed",
    "utilization",
]


# ==================================================================
# 5. HELPER
# ==================================================================

def print_section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ==================================================================
# 6. LOAD LOAN DATA
# ==================================================================

print_section("LOADING LOAN PORTFOLIO")

loan_df = pd.read_csv(
    LOAN_FILE,
    low_memory=False
)

loan_columns = [
    "loan_id",
    "defaulted",
    "sector",
    "loan_type",
    "collateral",
    "initial_rating",
    "credit_score",
    "ead",
    "coupon_rate",
    "leverage",
    "interest_coverage",
    "debt_to_equity",
    "maturity_months",
]

missing_loan_columns = [
    col
    for col in loan_columns
    if col not in loan_df.columns
]

if missing_loan_columns:

    raise ValueError(
        "Missing loan columns:\n"
        + "\n".join(missing_loan_columns)
    )

loan_df = loan_df[loan_columns].copy()

print(
    f"Loan rows: {len(loan_df):,}"
)


# ==================================================================
# 7. LOAD PAYMENT HISTORY
# ==================================================================

print_section("LOADING PAYMENT HISTORY")

payment_df = pd.read_csv(
    PAYMENT_FILE,
    low_memory=False
)

missing_payment_columns = [
    col
    for col in PAYMENT_COLUMNS
    if col not in payment_df.columns
]

if missing_payment_columns:

    raise ValueError(
        "Missing payment columns:\n"
        + "\n".join(missing_payment_columns)
    )

payment_df = payment_df[PAYMENT_COLUMNS].copy()

print(
    f"Payment rows: {len(payment_df):,}"
)


# ==================================================================
# 8. TYPE CONVERSION
# ==================================================================

print_section("CONVERTING DATA TYPES")

numeric_payment_columns = [
    "month_on_book",
    "scheduled_payment",
    "actual_payment",
    "outstanding_balance",
    "interest_component",
    "principal_component",
    "dpd",
    "missed_payment",
    "cumulative_missed",
    "utilization",
]

for col in numeric_payment_columns:

    payment_df[col] = pd.to_numeric(
        payment_df[col],
        errors="coerce"
    )

payment_df["payment_date"] = pd.to_datetime(
    payment_df["payment_date"],
    errors="coerce"
)

payment_df["loan_id"] = (
    payment_df["loan_id"]
    .astype("string")
    .str.strip()
)

loan_df["loan_id"] = (
    loan_df["loan_id"]
    .astype("string")
    .str.strip()
)


# ==================================================================
# 9. REMOVE EXACT DUPLICATES
# ==================================================================

print_section("PAYMENT DUPLICATE CHECK")

duplicate_count = payment_df.duplicated().sum()

print(
    f"Exact duplicate payment rows: "
    f"{duplicate_count:,}"
)

if duplicate_count > 0:

    payment_df = payment_df.drop_duplicates()

    print(
        f"Rows after duplicate removal: "
        f"{len(payment_df):,}"
    )


# ==================================================================
# 10. PAYMENT MONTH SANITY CHECK
# ==================================================================

print_section("PAYMENT MONTH SANITY CHECK")

print(
    payment_df["month_on_book"]
    .describe()
)

invalid_months = (
    payment_df["month_on_book"] < 1
).sum()

print(
    f"Invalid month_on_book values: "
    f"{invalid_months:,}"
)

if invalid_months > 0:

    payment_df = payment_df[
        payment_df["month_on_book"] >= 1
    ].copy()


# ==================================================================
# 11. KEEP ONLY LOANS IN LOAN PORTFOLIO
# ==================================================================

print_section("CHECKING LOAN ID MATCHING")

loan_ids = set(
    loan_df["loan_id"].dropna()
)

payment_df = payment_df[
    payment_df["loan_id"].isin(loan_ids)
].copy()

print(
    f"Payment records after loan-ID matching: "
    f"{len(payment_df):,}"
)


# ==================================================================
# 12. BEHAVIOURAL FEATURE FUNCTION
# ==================================================================

def create_window_features(
    payment_data,
    loan_data,
    months,
):
    """
    Create behavioural features using only the first N months.
    """

    print_section(
        f"CREATING {months}-MONTH FEATURES"
    )

    window_df = payment_data[
        payment_data["month_on_book"] <= months
    ].copy()

    print(
        f"Payment rows inside {months}-month window: "
        f"{len(window_df):,}"
    )

    # --------------------------------------------------------------
    # Derived payment variables
    # --------------------------------------------------------------

    window_df["payment_ratio"] = np.where(
        window_df["scheduled_payment"] > 0,
        window_df["actual_payment"]
        / window_df["scheduled_payment"],
        np.nan
    )

    window_df["payment_shortfall"] = (
        window_df["scheduled_payment"]
        - window_df["actual_payment"]
    )

    window_df["payment_shortfall"] = (
        window_df["payment_shortfall"]
        .clip(lower=0)
    )

    window_df["positive_payment"] = (
        window_df["actual_payment"] > 0
    ).astype(int)

    # --------------------------------------------------------------
    # Aggregate behavioural features
    # --------------------------------------------------------------

    grouped = (
        window_df
        .groupby("loan_id")
        .agg(
            payment_observations=(
                "month_on_book",
                "count"
            ),

            months_observed=(
                "month_on_book",
                "nunique"
            ),

            avg_dpd=(
                "dpd",
                "mean"
            ),

            max_dpd=(
                "dpd",
                "max"
            ),

            dpd_std=(
                "dpd",
                "std"
            ),

            avg_payment_ratio=(
                "payment_ratio",
                "mean"
            ),

            min_payment_ratio=(
                "payment_ratio",
                "min"
            ),

            payment_ratio_std=(
                "payment_ratio",
                "std"
            ),

            avg_payment_shortfall=(
                "payment_shortfall",
                "mean"
            ),

            total_payment_shortfall=(
                "payment_shortfall",
                "sum"
            ),

            missed_payment_count=(
                "missed_payment",
                "sum"
            ),

            avg_cumulative_missed=(
                "cumulative_missed",
                "mean"
            ),

            max_cumulative_missed=(
                "cumulative_missed",
                "max"
            ),

            avg_outstanding_balance=(
                "outstanding_balance",
                "mean"
            ),

            ending_outstanding_balance=(
                "outstanding_balance",
                "last"
            ),

            avg_utilization=(
                "utilization",
                "mean"
            ),

            max_utilization=(
                "utilization",
                "max"
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------------
    # Missing-value-safe derived features
    # --------------------------------------------------------------

    grouped["missed_payment_rate"] = (
        grouped["missed_payment_count"]
        / grouped["payment_observations"]
    )

    grouped["observation_coverage"] = (
        grouped["months_observed"]
        / months
    )

    grouped["dpd_30_plus_flag"] = (
        grouped["max_dpd"] >= 30
    ).astype(int)

    grouped["dpd_60_plus_flag"] = (
        grouped["max_dpd"] >= 60
    ).astype(int)

    grouped["dpd_90_plus_flag"] = (
        grouped["max_dpd"] >= 90
    ).astype(int)

    # --------------------------------------------------------------
    # Rename behavioural columns
    # --------------------------------------------------------------

    rename_map = {}

    for col in grouped.columns:

        if col == "loan_id":
            continue

        rename_map[col] = (
            f"{col}_{months}m"
        )

    grouped = grouped.rename(
        columns=rename_map
    )

    # --------------------------------------------------------------
    # Merge with loan-level information
    # --------------------------------------------------------------

    output = loan_data.merge(
        grouped,
        on="loan_id",
        how="inner"
    )

    print(
        f"Final {months}-month dataset: "
        f"{output.shape}"
    )

    return output


# ==================================================================
# 13. CREATE 3-MONTH DATASET
# ==================================================================

model_b_3m = create_window_features(
    payment_df,
    loan_df,
    months=3
)


# ==================================================================
# 14. CREATE 6-MONTH DATASET
# ==================================================================

model_b_6m = create_window_features(
    payment_df,
    loan_df,
    months=6
)


# ==================================================================
# 15. DATASET COVERAGE REPORT
# ==================================================================

print_section("MODEL B COVERAGE")

coverage_df = pd.DataFrame({
    "dataset": [
        "loan_portfolio",
        "model_b_3m",
        "model_b_6m"
    ],
    "rows": [
        len(loan_df),
        len(model_b_3m),
        len(model_b_6m)
    ]
})

coverage_df["coverage_percentage"] = (
    coverage_df["rows"]
    / len(loan_df)
    * 100
)

print(
    coverage_df.to_string(index=False)
)

coverage_df.to_csv(
    REPORT_DIR / "model_b_coverage.csv",
    index=False
)


# ==================================================================
# 16. MODEL B TARGET DISTRIBUTION
# ==================================================================

print_section("MODEL B TARGET DISTRIBUTION")

for name, data in [
    ("3m", model_b_3m),
    ("6m", model_b_6m),
]:

    distribution = (
        data["defaulted"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    distribution.columns = [
        "defaulted",
        "count"
    ]

    distribution["percentage"] = (
        distribution["count"]
        / len(data)
        * 100
    )

    print(
        f"\n{name.upper()}:"
    )

    print(
        distribution.to_string(index=False)
    )

    distribution.to_csv(
        REPORT_DIR
        / f"model_b_{name}_target_distribution.csv",
        index=False
    )


# ==================================================================
# 17. BEHAVIOURAL FEATURE REPORT
# ==================================================================

behavioural_prefixes = [
    "avg_dpd",
    "max_dpd",
    "dpd_std",
    "avg_payment_ratio",
    "min_payment_ratio",
    "payment_ratio_std",
    "avg_payment_shortfall",
    "total_payment_shortfall",
    "missed_payment_count",
    "avg_cumulative_missed",
    "max_cumulative_missed",
    "avg_outstanding_balance",
    "ending_outstanding_balance",
    "avg_utilization",
    "max_utilization",
    "missed_payment_rate",
    "observation_coverage",
    "dpd_30_plus_flag",
    "dpd_60_plus_flag",
    "dpd_90_plus_flag",
]


behavioural_features = []

for prefix in behavioural_prefixes:

    for months in [3, 6]:

        feature = f"{prefix}_{months}m"

        behavioural_features.append({
            "feature": feature,
            "window_months": months,
            "source": "payment_history",
            "role": "early_warning_predictor"
        })

behavioural_feature_df = pd.DataFrame(
    behavioural_features
)

behavioural_feature_df.to_csv(
    REPORT_DIR / "model_b_behavioural_features.csv",
    index=False
)


# ==================================================================
# 18. SAVE DATASETS
# ==================================================================

print_section("SAVING MODEL B DATASETS")

model_b_3m.to_parquet(
    OUTPUT_3M,
    index=False
)

model_b_6m.to_parquet(
    OUTPUT_6M,
    index=False
)

print(
    f"Saved 3-month dataset:\n{OUTPUT_3M}"
)

print(
    f"Saved 6-month dataset:\n{OUTPUT_6M}"
)


# ==================================================================
# 19. FINAL SUMMARY
# ==================================================================

print_section("PHASE 2B COMPLETE")

print(
    f"""
3-Month Early Warning Dataset:
    Rows    : {len(model_b_3m):,}
    Columns : {len(model_b_3m.columns):,}

6-Month Early Warning Dataset:
    Rows    : {len(model_b_6m):,}
    Columns : {len(model_b_6m.columns):,}

Important:
    - Only first N months were used.
    - No full-history aggregation was used.
    - No ML model was trained.
    - No SMOTE was applied.
    - No scaling was applied.
    - Raw CSV files were not modified.
"""
)