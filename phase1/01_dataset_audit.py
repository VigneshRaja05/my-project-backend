"""
===============================================================
PHASE 1 - DATASET AUDIT & FEATURE STRATEGY
===============================================================

Project:
Explainable MSME-Oriented Loan Risk Prediction

Dataset:
Synthetic Credit Risk Dataset - 50K Loans, 10 Sectors

Phase 1 Objectives:
1. Audit all CSV datasets
2. Inspect schemas
3. Check missing values
4. Check duplicates
5. Analyze target distribution
6. Analyze numerical features
7. Analyze categorical features
8. Identify possible ID columns
9. Identify potential leakage columns
10. Analyze relationships between datasets
11. Analyze payment-history coverage
12. Generate feature decision report

IMPORTANT:
No ML model is trained in this phase.

===============================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================
# 1. PROJECT PATHS
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "phase1" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================
# 2. DATASET FILES
# =============================================================

DATASETS = {
    "loan_portfolio": "loan_portfolio.csv",
    "payment_history": "payment_history.csv",
    "credit_ratings": "credit_ratings.csv",
    "macro_stress_scenarios": "macro_stress_scenarios.csv",
    "portfolio_metrics": "portfolio_metrics.csv",
    "vintage_analysis": "vintage_analysis.csv",
    "payment_history_dictionary": "payment_history_dictionary.csv",
}


# =============================================================
# 3. HELPER FUNCTIONS
# =============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def memory_usage_mb(df):
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def detect_id_columns(df):
    """
    Detect columns that are likely identifiers.
    """
    id_columns = []

    for col in df.columns:

        col_lower = col.lower()

        if (
            col_lower == "id"
            or col_lower.endswith("_id")
            or "identifier" in col_lower
        ):
            id_columns.append(col)

    return id_columns


def detect_date_columns(df):
    """
    Detect likely date columns based on column names.
    """
    date_columns = []

    for col in df.columns:

        col_lower = col.lower()

        if any(
            keyword in col_lower
            for keyword in [
                "date",
                "timestamp",
                "month",
                "year"
            ]
        ):
            date_columns.append(col)

    return date_columns


def classify_column_dtype(series):
    """
    Simple semantic classification.
    """

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    return "categorical/text"


# =============================================================
# 4. LOAD DATASETS
# =============================================================

print_section("PHASE 1 - DATASET AUDIT")

print(f"Project root : {PROJECT_ROOT}")
print(f"Raw data dir : {RAW_DIR}")
print(f"Report dir   : {REPORT_DIR}")

dataframes = {}

for name, filename in DATASETS.items():

    file_path = RAW_DIR / filename

    if not file_path.exists():

        print(f"\n[WARNING] Missing file:")
        print(f"         {file_path}")

        continue

    print(f"\nLoading: {filename}")

    df = pd.read_csv(file_path, low_memory=False)

    dataframes[name] = df

    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns):,}")
    print(f"Memory     : {memory_usage_mb(df):.2f} MB")


# =============================================================
# 5. DATASET OVERVIEW
# =============================================================

print_section("DATASET OVERVIEW")

dataset_summary = []

for name, df in dataframes.items():

    dataset_summary.append({
        "dataset": name,
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(memory_usage_mb(df), 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_percentage": round(
            df.isna().sum().sum()
            / (df.shape[0] * df.shape[1])
            * 100,
            4
        )
    })

dataset_summary_df = pd.DataFrame(dataset_summary)

print(dataset_summary_df.to_string(index=False))

dataset_summary_df.to_csv(
    REPORT_DIR / "dataset_summary.csv",
    index=False
)


# =============================================================
# 6. SCHEMA REPORT
# =============================================================

print_section("SCHEMA ANALYSIS")

schema_records = []

for dataset_name, df in dataframes.items():

    for col in df.columns:

        series = df[col]

        schema_records.append({
            "dataset": dataset_name,
            "column": col,
            "dtype": str(series.dtype),
            "semantic_type": classify_column_dtype(series),
            "rows": len(df),
            "non_null": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "missing_pct": round(
                series.isna().mean() * 100,
                4
            ),
            "unique_values": int(series.nunique(dropna=True)),
            "unique_pct": round(
                series.nunique(dropna=True)
                / len(df)
                * 100,
                4
            )
        })

schema_df = pd.DataFrame(schema_records)

schema_df.to_csv(
    REPORT_DIR / "schema_report.csv",
    index=False
)

print(schema_df.to_string(index=False))


# =============================================================
# 7. MISSING VALUE REPORT
# =============================================================

print_section("MISSING VALUE ANALYSIS")

missing_records = []

for dataset_name, df in dataframes.items():

    for col in df.columns:

        missing_count = int(df[col].isna().sum())

        missing_records.append({
            "dataset": dataset_name,
            "column": col,
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_count / len(df) * 100,
                4
            )
        })

missing_df = pd.DataFrame(missing_records)

missing_df = missing_df.sort_values(
    by="missing_percentage",
    ascending=False
)

missing_df.to_csv(
    REPORT_DIR / "missing_values.csv",
    index=False
)

print(
    missing_df[
        missing_df["missing_count"] > 0
    ].to_string(index=False)
)


# =============================================================
# 8. DUPLICATE ANALYSIS
# =============================================================

print_section("DUPLICATE ANALYSIS")

duplicate_records = []

for dataset_name, df in dataframes.items():

    duplicate_records.append({
        "dataset": dataset_name,
        "total_rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_percentage": round(
            df.duplicated().mean() * 100,
            4
        )
    })

duplicate_df = pd.DataFrame(duplicate_records)

duplicate_df.to_csv(
    REPORT_DIR / "duplicates.csv",
    index=False
)

print(duplicate_df.to_string(index=False))


# =============================================================
# 9. LOAN PORTFOLIO - TARGET ANALYSIS
# =============================================================

if "loan_portfolio" in dataframes:

    loan_df = dataframes["loan_portfolio"]

    print_section("TARGET ANALYSIS - LOAN PORTFOLIO")

    TARGET = "defaulted"

    if TARGET in loan_df.columns:

        target_counts = loan_df[TARGET].value_counts(
            dropna=False
        ).sort_index()

        target_percent = (
            loan_df[TARGET]
            .value_counts(normalize=True, dropna=False)
            .sort_index()
            * 100
        )

        target_df = pd.DataFrame({
            "target_value": target_counts.index,
            "count": target_counts.values,
            "percentage": target_percent.values
        })

        print(target_df.to_string(index=False))

        target_df.to_csv(
            REPORT_DIR / "target_distribution.csv",
            index=False
        )

        if set(loan_df[TARGET].dropna().unique()).issubset(
            {0, 1}
        ):

            default_rate = loan_df[TARGET].mean() * 100

            print(
                f"\nDefault rate: {default_rate:.2f}%"
            )

    else:

        print(
            f"[WARNING] Target '{TARGET}' not found."
        )


# =============================================================
# 10. NUMERICAL FEATURE SUMMARY
# =============================================================

print_section("NUMERICAL FEATURE ANALYSIS")

if "loan_portfolio" in dataframes:

    loan_df = dataframes["loan_portfolio"]

    numerical_df = loan_df.select_dtypes(
        include=np.number
    )

    numerical_summary = numerical_df.describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99
        ]
    ).T

    numerical_summary["missing"] = (
        numerical_df.isna().sum()
    )

    numerical_summary["missing_pct"] = (
        numerical_df.isna().mean() * 100
    )

    numerical_summary.to_csv(
        REPORT_DIR / "numerical_summary.csv"
    )

    print(numerical_summary.to_string())


# =============================================================
# 11. CATEGORICAL FEATURE SUMMARY
# =============================================================

print_section("CATEGORICAL FEATURE ANALYSIS")

if "loan_portfolio" in dataframes:

    loan_df = dataframes["loan_portfolio"]

    categorical_columns = loan_df.select_dtypes(
        include=["object", "category"]
    ).columns

    categorical_records = []

    for col in categorical_columns:

        value_counts = (
            loan_df[col]
            .value_counts(dropna=False)
        )

        for value, count in value_counts.items():

            categorical_records.append({
                "column": col,
                "value": value,
                "count": int(count),
                "percentage": round(
                    count / len(loan_df) * 100,
                    4
                )
            })

    categorical_summary = pd.DataFrame(
        categorical_records
    )

    categorical_summary.to_csv(
        REPORT_DIR / "categorical_summary.csv",
        index=False
    )

    print(
        categorical_summary.to_string(index=False)
    )


# =============================================================
# 12. ID COLUMN DETECTION
# =============================================================

print_section("ID COLUMN DETECTION")

id_records = []

for dataset_name, df in dataframes.items():

    id_columns = detect_id_columns(df)

    for col in id_columns:

        id_records.append({
            "dataset": dataset_name,
            "column": col,
            "unique_values": int(
                df[col].nunique(dropna=True)
            ),
            "rows": len(df),
            "unique_percentage": round(
                df[col].nunique(dropna=True)
                / len(df)
                * 100,
                4
            )
        })

id_df = pd.DataFrame(id_records)

if len(id_df) > 0:

    print(id_df.to_string(index=False))

else:

    print("No obvious ID columns detected.")

id_df.to_csv(
    REPORT_DIR / "id_columns.csv",
    index=False
)


# =============================================================
# 13. DATE COLUMN DETECTION
# =============================================================

print_section("DATE COLUMN DETECTION")

date_records = []

for dataset_name, df in dataframes.items():

    date_columns = detect_date_columns(df)

    for col in date_columns:

        date_records.append({
            "dataset": dataset_name,
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "unique_values": int(
                df[col].nunique(dropna=True)
            )
        })

date_df = pd.DataFrame(date_records)

if len(date_df) > 0:

    print(date_df.to_string(index=False))

else:

    print("No obvious date columns detected.")

date_df.to_csv(
    REPORT_DIR / "date_columns.csv",
    index=False
)


# =============================================================
# 14. CORRELATION WITH TARGET
# =============================================================

print_section("NUMERICAL FEATURE vs TARGET CORRELATION")

if "loan_portfolio" in dataframes:

    loan_df = dataframes["loan_portfolio"]

    TARGET = "defaulted"

    numeric_cols = loan_df.select_dtypes(
        include=np.number
    ).columns

    if TARGET in numeric_cols:

        correlation = (
            loan_df[numeric_cols]
            .corr()[TARGET]
            .drop(TARGET)
            .sort_values(
                key=lambda x: abs(x),
                ascending=False
            )
        )

        correlation_df = (
            correlation
            .reset_index()
        )

        correlation_df.columns = [
            "feature",
            "correlation_with_default"
        ]

        correlation_df[
            "absolute_correlation"
        ] = correlation_df[
            "correlation_with_default"
        ].abs()

        print(
            correlation_df.to_string(index=False)
        )

        correlation_df.to_csv(
            REPORT_DIR / "target_correlations.csv",
            index=False
        )


# =============================================================
# 15. DEFAULT RATE BY CATEGORICAL VARIABLES
# =============================================================

print_section("DEFAULT RATE BY CATEGORICAL FEATURES")

if "loan_portfolio" in dataframes:

    loan_df = dataframes["loan_portfolio"]

    TARGET = "defaulted"

    categorical_columns = [
        col
        for col in loan_df.select_dtypes(
            include=["object", "category"]
        ).columns
        if col != "loan_id"
    ]

    default_rate_records = []

    for col in categorical_columns:

        grouped = (
            loan_df
            .groupby(col, dropna=False)[TARGET]
            .agg(
                loans="count",
                defaults="sum",
                default_rate="mean"
            )
            .reset_index()
        )

        grouped["feature"] = col
        grouped["category"] = grouped[col].astype(str)

        grouped["default_rate"] = (
            grouped["default_rate"] * 100
        )

        grouped = grouped[
            [
                "feature",
                "category",
                "loans",
                "defaults",
                "default_rate"
            ]
        ]

        default_rate_records.append(grouped)

    if default_rate_records:

        default_rate_df = pd.concat(
            default_rate_records,
            ignore_index=True
        )

        default_rate_df = default_rate_df.sort_values(
            by=["feature", "default_rate"],
            ascending=[True, False]
        )

        default_rate_df.to_csv(
            REPORT_DIR / "categorical_default_rates.csv",
            index=False
        )

        print(
            default_rate_df.to_string(index=False)
        )

# =============================================================
# 16. PAYMENT HISTORY ANALYSIS
# =============================================================

print_section("PAYMENT HISTORY ANALYSIS")

if (
    "payment_history" in dataframes
    and "loan_portfolio" in dataframes
):

    payment_df = dataframes["payment_history"]
    loan_df = dataframes["loan_portfolio"]

    print(
        f"Payment records: {len(payment_df):,}"
    )

    if "loan_id" in payment_df.columns:

        payment_loan_ids = set(
            payment_df["loan_id"]
            .dropna()
            .unique()
        )

        loan_ids = set(
            loan_df["loan_id"]
            .dropna()
            .unique()
        )

        matching_ids = payment_loan_ids.intersection(
            loan_ids
        )

        print(
            f"Unique loans in payment history: "
            f"{len(payment_loan_ids):,}"
        )

        print(
            f"Matching loan IDs: "
            f"{len(matching_ids):,}"
        )

        coverage = (
            len(matching_ids)
            / len(loan_ids)
            * 100
        )

        print(
            f"Payment-history coverage: "
            f"{coverage:.2f}%"
        )

        payment_coverage_df = pd.DataFrame({
            "metric": [
                "total_loans",
                "loans_with_payment_history",
                "coverage_percentage"
            ],
            "value": [
                len(loan_ids),
                len(matching_ids),
                coverage
            ]
        })

        payment_coverage_df.to_csv(
            REPORT_DIR / "payment_history_coverage.csv",
            index=False
        )

        # -----------------------------------------------------
        # Number of payment records per loan
        # -----------------------------------------------------

        records_per_loan = (
            payment_df
            .groupby("loan_id")
            .size()
            .reset_index(
                name="payment_record_count"
            )
        )

        records_per_loan.to_csv(
            REPORT_DIR / "payment_records_per_loan.csv",
            index=False
        )

        print(
            "\nPayment records per loan:"
        )

        print(
            records_per_loan[
                "payment_record_count"
            ].describe().to_string()
        )


# =============================================================
# 17. PAYMENT HISTORY TARGET RELATIONSHIP
# =============================================================

if (
    "payment_history" in dataframes
    and "loan_portfolio" in dataframes
):

    payment_df = dataframes["payment_history"]
    loan_df = dataframes["loan_portfolio"]

    if (
        "loan_id" in payment_df.columns
        and "loan_id" in loan_df.columns
        and "defaulted" in loan_df.columns
    ):

        payment_loan_counts = (
            payment_df
            .groupby("loan_id")
            .size()
            .reset_index(
                name="payment_records"
            )
        )

        payment_target = payment_loan_counts.merge(
            loan_df[
                [
                    "loan_id",
                    "defaulted"
                ]
            ],
            on="loan_id",
            how="left"
        )

        payment_target_summary = (
            payment_target
            .groupby("defaulted")[
                "payment_records"
            ]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "min",
                    "max"
                ]
            )
            .reset_index()
        )

        payment_target_summary.to_csv(
            REPORT_DIR
            / "payment_history_by_target.csv",
            index=False
        )

        print_section(
            "PAYMENT HISTORY vs DEFAULT STATUS"
        )

        print(
            payment_target_summary.to_string(
                index=False
            )
        )


# =============================================================
# 18. PRELIMINARY FEATURE DECISION TABLE
# =============================================================

print_section("FEATURE DECISION TABLE")

feature_decisions = [

    # ---------------------------------------------------------
    # KEEP
    # ---------------------------------------------------------

    {
        "feature": "sector",
        "source": "loan_portfolio",
        "decision": "KEEP",
        "reason": "Sector-level risk information",
        "model": "A + B"
    },

    {
        "feature": "loan_type",
        "source": "loan_portfolio",
        "decision": "KEEP",
        "reason": "Loan characteristics",
        "model": "A + B"
    },

    {
        "feature": "collateral",
        "source": "loan_portfolio",
        "decision": "KEEP",
        "reason": "Collateral information available at origination",
        "model": "A + B"
    },

    {
        "feature": "initial_rating",
        "source": "loan_portfolio",
        "decision": "KEEP",
        "reason": "Initial credit assessment",
        "model": "A + B"
    },

    {
        "feature": "credit_score",
        "source": "loan_portfolio",
        "decision": "KEEP",
        "reason": "Credit quality indicator",
        "model": "A + B"
    },

    {
        "feature": "ead",
        "source": "loan_portfolio",
        "decision": "KEEP",
        "reason": "Exposure at origination",
        "model": "A + B"
    },

    {
        "feature": "coupon_rate",
        "source": "loan_portfolio",
        "decision": "KEEP",
        "reason": "Loan pricing characteristic",
        "model": "A + B"
    },

    {
        "feature": "leverage",
        "source": "loan_portfolio",
        "decision": "KEEP",
        "reason": "Financial leverage indicator",
        "model": "A + B"
    },

    {
        "feature": "interest_coverage",
        "source": "loan_portfolio",
        "decision": "KEEP",
        "reason": "Debt servicing capacity indicator",
        "model": "A + B"
    },

    {
        "feature": "debt_to_equity",
        "source": "loan_portfolio",
        "decision": "KEEP",
        "reason": "Capital structure indicator",
        "model": "A + B"
    },

    {
        "feature": "maturity_months",
        "source": "loan_portfolio",
        "decision": "KEEP",
        "reason": "Loan term information",
        "model": "A + B"
    },

    # ---------------------------------------------------------
    # EXCLUDE / REMOVE
    # ---------------------------------------------------------

    {
        "feature": "loan_id",
        "source": "loan_portfolio",
        "decision": "REMOVE",
        "reason": "Identifier only",
        "model": "None"
    },

    {
        "feature": "default_date",
        "source": "loan_portfolio",
        "decision": "REMOVE",
        "reason": "Post-outcome information / target leakage",
        "model": "None"
    },

    {
        "feature": "survival_months",
        "source": "loan_portfolio",
        "decision": "REMOVE",
        "reason": "Outcome-period information / target leakage",
        "model": "None"
    },

    {
        "feature": "recovery_rate",
        "source": "loan_portfolio",
        "decision": "REMOVE",
        "reason": "Post-default information",
        "model": "None"
    },

    {
        "feature": "loss_given_default",
        "source": "loan_portfolio",
        "decision": "REMOVE",
        "reason": "Post-default information",
        "model": "None"
    },

    {
        "feature": "pd_annual",
        "source": "loan_portfolio",
        "decision": "EXCLUDE_INITIAL",
        "reason": "Existing probability-of-default estimate; investigate as benchmark",
        "model": "Benchmark"
    },

    {
        "feature": "lgd",
        "source": "loan_portfolio",
        "decision": "EXCLUDE_INITIAL",
        "reason": "Risk-management derived quantity; investigate separately",
        "model": "Analysis"
    },

    {
        "feature": "el",
        "source": "loan_portfolio",
        "decision": "EXCLUDE_INITIAL",
        "reason": "Derived expected-loss measure; avoid circular modelling",
        "model": "Analysis"
    },

    {
        "feature": "unexpected_loss",
        "source": "loan_portfolio",
        "decision": "EXCLUDE_INITIAL",
        "reason": "Derived risk measure",
        "model": "Analysis"
    },

    {
        "feature": "rwa",
        "source": "loan_portfolio",
        "decision": "EXCLUDE_INITIAL",
        "reason": "Derived risk-weighted asset measure",
        "model": "Analysis"
    },

    # ---------------------------------------------------------
    # ENGINEER LATER
    # ---------------------------------------------------------

    {
        "feature": "payment_behaviour_features",
        "source": "payment_history",
        "decision": "ENGINEER",
        "reason": "Create early-warning indicators using only first N months",
        "model": "B"
    },

    {
        "feature": "macro_stress_features",
        "source": "macro_stress_scenarios",
        "decision": "ANALYSIS",
        "reason": "Use for economic stress testing",
        "model": "Stress Testing"
    },

    {
        "feature": "rating_migration",
        "source": "credit_ratings",
        "decision": "ANALYSIS",
        "reason": "Use for credit deterioration analysis",
        "model": "Analysis"
    },

    {
        "feature": "vintage_features",
        "source": "vintage_analysis",
        "decision": "ANALYSIS",
        "reason": "Use for cohort/default analysis",
        "model": "Analysis"
    },

    {
        "feature": "portfolio_metrics",
        "source": "portfolio_metrics",
        "decision": "ANALYSIS",
        "reason": "Portfolio-level risk analysis",
        "model": "Analysis"
    },
]

feature_decision_df = pd.DataFrame(
    feature_decisions
)

feature_decision_df.to_csv(
    REPORT_DIR / "feature_decision.csv",
    index=False
)

print(
    feature_decision_df.to_string(index=False)
)


# =============================================================
# 19. FINAL AUDIT SUMMARY
# =============================================================

print_section("PHASE 1 AUDIT COMPLETE")

print(
    """
PHASE 1 OUTPUTS GENERATED:

1. dataset_summary.csv
2. schema_report.csv
3. missing_values.csv
4. duplicates.csv
5. target_distribution.csv
6. numerical_summary.csv
7. categorical_summary.csv
8. id_columns.csv
9. date_columns.csv
10. target_correlations.csv
11. categorical_default_rates.csv
12. payment_history_coverage.csv
13. payment_records_per_loan.csv
14. payment_history_by_target.csv
15. feature_decision.csv

IMPORTANT:
No machine-learning model was trained.

NEXT PHASE:
Phase 2 - Data Preprocessing & Feature Engineering
"""
)

print("\nAudit completed successfully.")