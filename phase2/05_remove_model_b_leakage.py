from pathlib import Path
import pandas as pd


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)


FILE_3M = (
    PROCESSED_DIR
    / "model_b_final_early_warning_3m.parquet"
)

FILE_6M = (
    PROCESSED_DIR
    / "model_b_final_early_warning_6m.parquet"
)


# ================================================================
# PROCESS FUNCTION
# ================================================================

def clean_model_b(
    input_file,
    target_column
):

    print("\n" + "=" * 80)
    print(f"PROCESSING: {input_file.name}")
    print("=" * 80)

    df = pd.read_parquet(input_file)

    print(
        f"Original shape: {df.shape}"
    )

    # ------------------------------------------------------------
    # Remove original outcome label
    # ------------------------------------------------------------

    leakage_columns = [
        "defaulted",
        "defaulted_portfolio",
        "default_date",
        "months_to_default",
        "days_to_default"
    ]

    existing = [
        col
        for col in leakage_columns
        if col in df.columns
    ]

    print(
        f"Removing leakage columns: {existing}"
    )

    df = df.drop(
        columns=existing,
        errors="ignore"
    )

    # ------------------------------------------------------------
    # Verify target still exists
    # ------------------------------------------------------------

    if target_column not in df.columns:

        raise ValueError(
            f"Target column missing: {target_column}"
        )

    # ------------------------------------------------------------
    # Verify no leakage columns remain
    # ------------------------------------------------------------

    forbidden = [
        col
        for col in leakage_columns
        if col in df.columns
    ]

    if forbidden:

        raise ValueError(
            f"Leakage columns still present: {forbidden}"
        )

    # ------------------------------------------------------------
    # Duplicate check
    # ------------------------------------------------------------

    duplicate_ids = (
        df["loan_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids > 0:

        raise ValueError(
            f"Duplicate loan IDs: {duplicate_ids}"
        )

    # ------------------------------------------------------------
    # Missing target check
    # ------------------------------------------------------------

    missing_target = (
        df[target_column]
        .isna()
        .sum()
    )

    if missing_target > 0:

        raise ValueError(
            f"Missing target values: {missing_target}"
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    df.to_parquet(
        input_file,
        index=False
    )

    print(
        f"Final shape: {df.shape}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        df[target_column]
        .value_counts()
        .sort_index()
    )

    print(
        "\nRemaining columns:"
    )

    print(
        df.columns.tolist()
    )

    print(
        "\nSaved successfully."
    )


# ================================================================
# 3M
# ================================================================

clean_model_b(
    FILE_3M,
    "early_warning_target_3m"
)


# ================================================================
# 6M
# ================================================================

clean_model_b(
    FILE_6M,
    "early_warning_target_6m"
)


# ================================================================
# COMPLETE
# ================================================================

print("\n" + "=" * 80)
print("MODEL B LEAKAGE CLEANUP COMPLETE")
print("=" * 80)

print(
    """
3M and 6M Model B datasets are now ready for Phase 3.

Removed:
    - defaulted
    - defaulted_portfolio
    - default_date
    - months_to_default
    - days_to_default

No ML model was trained.
"""
)