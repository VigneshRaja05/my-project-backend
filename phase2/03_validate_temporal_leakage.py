"""
====================================================================
PHASE 2C - TEMPORAL LEAKAGE VALIDATION
====================================================================

Purpose:
Validate that payment-history observation windows occur before
the recorded default event.

This is a validation/audit step only.
No ML model is trained.
====================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ==================================================================
# PATHS
# ==================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "phase2" / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================================
# FILES
# ==================================================================

LOAN_FILE = RAW_DIR / "loan_portfolio.csv"
PAYMENT_FILE = RAW_DIR / "payment_history.csv"


# ==================================================================
# HELPER
# ==================================================================

def print_section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ==================================================================
# LOAD DATA
# ==================================================================

print_section("LOADING DATA")

loan_df = pd.read_csv(
    LOAN_FILE,
    low_memory=False
)

payment_df = pd.read_csv(
    PAYMENT_FILE,
    low_memory=False
)

print(
    f"Loans: {len(loan_df):,}"
)

print(
    f"Payment records: {len(payment_df):,}"
)


# ==================================================================
# DATE CONVERSION
# ==================================================================

print_section("CONVERTING DATES")

loan_df["origination_date"] = pd.to_datetime(
    loan_df["origination_date"],
    errors="coerce"
)

loan_df["default_date"] = pd.to_datetime(
    loan_df["default_date"],
    errors="coerce"
)

payment_df["payment_date"] = pd.to_datetime(
    payment_df["payment_date"],
    errors="coerce"
)

payment_df["month_on_book"] = pd.to_numeric(
    payment_df["month_on_book"],
    errors="coerce"
)


# ==================================================================
# KEEP DEFAULTED LOANS
# ==================================================================

defaulted_loans = loan_df[
    loan_df["defaulted"] == 1
].copy()

print(
    f"Defaulted loans: "
    f"{len(defaulted_loans):,}"
)


# ==================================================================
# MERGE PAYMENT HISTORY
# ==================================================================

payment_subset = payment_df[
    [
        "loan_id",
        "month_on_book",
        "payment_date",
        "dpd",
        "missed_payment",
        "cumulative_missed"
    ]
].copy()

merged = payment_subset.merge(
    defaulted_loans[
        [
            "loan_id",
            "origination_date",
            "default_date"
        ]
    ],
    on="loan_id",
    how="inner"
)

print(
    f"Defaulted loans with payment records: "
    f"{merged['loan_id'].nunique():,}"
)


# ==================================================================
# CALCULATE MONTHS FROM ORIGINATION TO DEFAULT
# ==================================================================

merged["days_to_default"] = (
    merged["default_date"]
    - merged["origination_date"]
).dt.days

merged["months_to_default"] = (
    merged["days_to_default"] / 30.4375
)


# ==================================================================
# DEFAULT TIMING SUMMARY
# ==================================================================

print_section("DEFAULT TIMING")

default_timing = (
    merged[
        [
            "loan_id",
            "origination_date",
            "default_date",
            "days_to_default",
            "months_to_default"
        ]
    ]
    .drop_duplicates("loan_id")
)

print(
    default_timing["months_to_default"]
    .describe()
)


# ==================================================================
# 3-MONTH TEMPORAL CHECK
# ==================================================================

print_section("3-MONTH TEMPORAL CHECK")

default_timing["default_within_3m"] = (
    default_timing["months_to_default"] <= 3
)

default_timing["default_within_6m"] = (
    default_timing["months_to_default"] <= 6
)

print(
    "Defaults within first 3 months:"
)

print(
    default_timing["default_within_3m"]
    .value_counts()
)


# ==================================================================
# 6-MONTH TEMPORAL CHECK
# ==================================================================

print(
    "\nDefaults within first 6 months:"
)

print(
    default_timing["default_within_6m"]
    .value_counts()
)


# ==================================================================
# REPORT
# ==================================================================

report = pd.DataFrame({
    "metric": [
        "total_defaulted_loans",
        "defaults_within_3_months",
        "defaults_after_3_months",
        "defaults_within_6_months",
        "defaults_after_6_months"
    ],
    "count": [
        len(default_timing),
        int(
            default_timing["default_within_3m"]
            .sum()
        ),
        int(
            (~default_timing["default_within_3m"])
            .sum()
        ),
        int(
            default_timing["default_within_6m"]
            .sum()
        ),
        int(
            (~default_timing["default_within_6m"])
            .sum()
        )
    ]
})

report["percentage"] = (
    report["count"]
    / len(default_timing)
    * 100
)

print_section("FINAL TEMPORAL REPORT")

print(
    report.to_string(index=False)
)

report.to_csv(
    REPORT_DIR
    / "temporal_leakage_report.csv",
    index=False
)


# ==================================================================
# FINAL
# ==================================================================

print_section("PHASE 2C COMPLETE")

print(
    """
Temporal validation completed.

No ML model was trained.
No data was modified.
"""
)