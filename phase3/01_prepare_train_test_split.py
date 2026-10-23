"""
====================================================================
PHASE 3.1 - TRAIN / VALIDATION / TEST SPLIT + PREPROCESSING SETUP
====================================================================

Project:
Predicting Loan Risk of MSME Using Alternative Financial Data

Purpose:
    1. Load finalized Phase 2 datasets.
    2. Remove identifiers from ML features.
    3. Separate X and y.
    4. Create stratified train/validation/test splits.
    5. Build preprocessing pipelines.
    6. Fit preprocessing ONLY on training data.
    7. Transform validation and test data.
    8. Save split datasets and preprocessing objects.

NO MODEL IS TRAINED IN THIS PHASE.

Datasets:
    Model A:
        Origination default-risk prediction

    Model B-3M:
        3-month early-warning prediction

    Model B-6M:
        6-month early-warning prediction

Split:
    70% development data
    30% test data

Development data:
    80% train
    20% validation

Final:
    56% train
    14% validation
    30% test

Stratification:
    Target class distribution is preserved.

Important:
    - loan_id is NOT used as a predictor.
    - Preprocessors are fitted ONLY on training data.
    - Test data remains completely unseen.
====================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer


# ==================================================================
# 1. PROJECT PATHS
# ==================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

PHASE3_DIR = (
    PROJECT_ROOT / "data" / "phase3"
)

SPLIT_DIR = (
    PHASE3_DIR / "splits"
)

PREPROCESSOR_DIR = (
    PHASE3_DIR / "preprocessors"
)

REPORT_DIR = (
    PROJECT_ROOT / "phase3" / "reports"
)


# Create directories

for directory in [
    PHASE3_DIR,
    SPLIT_DIR,
    PREPROCESSOR_DIR,
    REPORT_DIR
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ==================================================================
# 2. INPUT FILES
# ==================================================================

DATASETS = {

    "model_a": {
        "file":
            PROCESSED_DIR
            / "model_a_origination_dataset.parquet",

        "target":
            "defaulted"
    },

    "model_b_3m": {
        "file":
            PROCESSED_DIR
            / "model_b_final_early_warning_3m.parquet",

        "target":
            "early_warning_target_3m"
    },

    "model_b_6m": {
        "file":
            PROCESSED_DIR
            / "model_b_final_early_warning_6m.parquet",

        "target":
            "early_warning_target_6m"
    }
}


# ==================================================================
# 3. RANDOM STATE
# ==================================================================

RANDOM_STATE = 42


# ==================================================================
# 4. HELPER FUNCTION
# ==================================================================

def print_section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ==================================================================
# 5. DATASET PROCESSING FUNCTION
# ==================================================================

def process_dataset(
    dataset_name,
    config
):

    target_column = config["target"]
    input_file = config["file"]

    print_section(
        f"PROCESSING {dataset_name.upper()}"
    )

    # --------------------------------------------------------------
    # Check file
    # --------------------------------------------------------------

    if not input_file.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{input_file}"
        )

    # --------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------

    df = pd.read_parquet(
        input_file
    )

    print(
        f"Input file : {input_file.name}"
    )

    print(
        f"Rows       : {len(df):,}"
    )

    print(
        f"Columns    : {len(df.columns):,}"
    )

    # --------------------------------------------------------------
    # Verify target
    # --------------------------------------------------------------

    if target_column not in df.columns:

        raise ValueError(
            f"Target column not found: "
            f"{target_column}"
        )

    # --------------------------------------------------------------
    # Verify loan_id
    # --------------------------------------------------------------

    if "loan_id" not in df.columns:

        raise ValueError(
            "loan_id column is missing."
        )

    # --------------------------------------------------------------
    # Missing-value report
    # --------------------------------------------------------------

    print_section(
        f"{dataset_name.upper()} MISSING VALUE CHECK"
    )

    missing_df = pd.DataFrame({
        "feature":
            df.columns,

        "missing_count":
            [
                int(df[col].isna().sum())
                for col in df.columns
            ],

        "missing_percentage":
            [
                float(
                    df[col].isna().mean() * 100
                )
                for col in df.columns
            ]
    })

    missing_features = (
        missing_df[
            missing_df["missing_count"] > 0
        ]
    )

    if len(missing_features) == 0:

        print(
            "No missing values detected."
        )

    else:

        print(
            missing_features.to_string(
                index=False
            )
        )

    missing_df.to_csv(
        REPORT_DIR
        / f"{dataset_name}_missing_values.csv",
        index=False
    )

    # --------------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------------

    print_section(
        f"{dataset_name.upper()} DUPLICATE CHECK"
    )

    duplicate_ids = (
        df["loan_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate loan IDs: "
        f"{duplicate_ids:,}"
    )

    if duplicate_ids > 0:

        raise ValueError(
            "Duplicate loan IDs detected."
        )

    # --------------------------------------------------------------
    # Target validation
    # --------------------------------------------------------------

    print_section(
        f"{dataset_name.upper()} TARGET DISTRIBUTION"
    )

    target_distribution = (
        df[target_column]
        .value_counts()
        .sort_index()
    )

    target_percentage = (
        df[target_column]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    target_report = pd.DataFrame({
        "class":
            target_distribution.index,

        "count":
            target_distribution.values,

        "percentage":
            target_percentage.values
    })

    print(
        target_report.to_string(
            index=False
        )
    )

    target_report.to_csv(
        REPORT_DIR
        / f"{dataset_name}_target_distribution.csv",
        index=False
    )

    # --------------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------------

    X = df.drop(
        columns=[target_column]
    ).copy()

    y = df[
        target_column
    ].astype(int).copy()

    # --------------------------------------------------------------
    # Remove identifier from predictors
    # --------------------------------------------------------------

    X_model = X.drop(
        columns=["loan_id"]
    )

    print_section(
        f"{dataset_name.upper()} FEATURE SEPARATION"
    )

    print(
        f"Original columns : "
        f"{len(df.columns)}"
    )

    print(
        f"Target           : "
        f"{target_column}"
    )

    print(
        f"Identifier       : "
        f"loan_id"
    )

    print(
        f"Predictor count  : "
        f"{len(X_model.columns)}"
    )

    # --------------------------------------------------------------
    # Identify categorical and numerical features
    # --------------------------------------------------------------

    categorical_features = (
        X_model
        .select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    numerical_features = (
        X_model
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    print(
        f"\nCategorical features: "
        f"{len(categorical_features)}"
    )

    for col in categorical_features:

        print(
            f"    {col}"
        )

    print(
        f"\nNumerical features: "
        f"{len(numerical_features)}"
    )

    for col in numerical_features:

        print(
            f"    {col}"
        )

    # --------------------------------------------------------------
    # Verify every feature has a type
    # --------------------------------------------------------------

    unclassified = [
        col
        for col in X_model.columns
        if (
            col not in categorical_features
            and
            col not in numerical_features
        )
    ]

    if unclassified:

        raise ValueError(
            "Unclassified features found:\n"
            + "\n".join(unclassified)
        )

    # --------------------------------------------------------------
    # TRAIN / TEMPORARY TEST SPLIT
    # --------------------------------------------------------------

    print_section(
        f"{dataset_name.upper()} TRAIN / TEST SPLIT"
    )

    X_dev, X_test, y_dev, y_test = (
        train_test_split(
            X_model,
            y,
            test_size=0.30,
            stratify=y,
            random_state=RANDOM_STATE
        )
    )

    print(
        f"Development rows: "
        f"{len(X_dev):,}"
    )

    print(
        f"Test rows       : "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------------

    print_section(
        f"{dataset_name.upper()} TRAIN / VALIDATION SPLIT"
    )

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            X_dev,
            y_dev,
            test_size=0.20,
            stratify=y_dev,
            random_state=RANDOM_STATE
        )
    )

    print(
        f"Training rows   : "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows : "
        f"{len(X_valid):,}"
    )

    print(
        f"Test rows       : "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------------
    # Verify split sizes
    # --------------------------------------------------------------

    total_rows = (
        len(X_train)
        + len(X_valid)
        + len(X_test)
    )

    if total_rows != len(df):

        raise ValueError(
            "Split row count does not match "
            "original dataset."
        )

    # --------------------------------------------------------------
    # Verify no loan overlap
    # --------------------------------------------------------------

    # loan_id is no longer inside X_model,
    # so use original dataframe indices.

    train_indices = X_train.index
    valid_indices = X_valid.index
    test_indices = X_test.index

    train_set = set(train_indices)
    valid_set = set(valid_indices)
    test_set = set(test_indices)

    if train_set & valid_set:

        raise ValueError(
            "Train/validation overlap detected."
        )

    if train_set & test_set:

        raise ValueError(
            "Train/test overlap detected."
        )

    if valid_set & test_set:

        raise ValueError(
            "Validation/test overlap detected."
        )

    print(
        "\nSplit overlap check: PASSED"
    )

    # --------------------------------------------------------------
    # Stratification verification
    # --------------------------------------------------------------

    print_section(
        f"{dataset_name.upper()} CLASS DISTRIBUTION BY SPLIT"
    )

    split_distribution = []

    for split_name, target_values in [
        ("train", y_train),
        ("validation", y_valid),
        ("test", y_test)
    ]:

        counts = (
            target_values
            .value_counts()
            .sort_index()
        )

        percentages = (
            target_values
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        )

        for class_value in counts.index:

            split_distribution.append({
                "dataset":
                    dataset_name,

                "split":
                    split_name,

                "class":
                    int(class_value),

                "count":
                    int(counts[class_value]),

                "percentage":
                    float(
                        percentages[class_value]
                    )
            })

    split_report = pd.DataFrame(
        split_distribution
    )

    print(
        split_report.to_string(
            index=False
        )
    )

    split_report.to_csv(
        REPORT_DIR
        / f"{dataset_name}_split_distribution.csv",
        index=False
    )

    # --------------------------------------------------------------
    # Create preprocessing pipelines
    # --------------------------------------------------------------

    print_section(
        f"{dataset_name.upper()} PREPROCESSING PIPELINE"
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    # --------------------------------------------------------------
    # FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------------

    print(
        "\nFitting preprocessor ONLY on training data..."
    )

    preprocessor.fit(
        X_train
    )

    print(
        "Preprocessor fitted successfully."
    )

    # --------------------------------------------------------------
    # Transform datasets
    # --------------------------------------------------------------

    print(
        "\nTransforming datasets..."
    )

    X_train_processed = (
        preprocessor.transform(
            X_train
        )
    )

    X_valid_processed = (
        preprocessor.transform(
            X_valid
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        f"Processed training shape   : "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed validation shape : "
        f"{X_valid_processed.shape}"
    )

    print(
        f"Processed test shape       : "
        f"{X_test_processed.shape}"
    )

    # --------------------------------------------------------------
    # Get transformed feature names
    # --------------------------------------------------------------

    transformed_feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    print(
        f"\nTransformed feature count: "
        f"{len(transformed_feature_names)}"
    )

    # --------------------------------------------------------------
    # Convert processed arrays to DataFrames
    # --------------------------------------------------------------

    X_train_processed_df = pd.DataFrame(
        X_train_processed,
        columns=transformed_feature_names,
        index=X_train.index
    )

    X_valid_processed_df = pd.DataFrame(
        X_valid_processed,
        columns=transformed_feature_names,
        index=X_valid.index
    )

    X_test_processed_df = pd.DataFrame(
        X_test_processed,
        columns=transformed_feature_names,
        index=X_test.index
    )

    # --------------------------------------------------------------
    # Save raw splits
    # --------------------------------------------------------------

    print_section(
        f"{dataset_name.upper()} SAVING RAW SPLITS"
    )

    train_raw = X_train.copy()

    train_raw[target_column] = y_train

    valid_raw = X_valid.copy()

    valid_raw[target_column] = y_valid

    test_raw = X_test.copy()

    test_raw[target_column] = y_test

    train_raw.to_parquet(
        SPLIT_DIR
        / f"{dataset_name}_train_raw.parquet",
        index=True
    )

    valid_raw.to_parquet(
        SPLIT_DIR
        / f"{dataset_name}_validation_raw.parquet",
        index=True
    )

    test_raw.to_parquet(
        SPLIT_DIR
        / f"{dataset_name}_test_raw.parquet",
        index=True
    )

    # --------------------------------------------------------------
    # Save processed splits
    # --------------------------------------------------------------

    X_train_processed_df.to_parquet(
        SPLIT_DIR
        / f"{dataset_name}_X_train.parquet",
        index=True
    )

    X_valid_processed_df.to_parquet(
        SPLIT_DIR
        / f"{dataset_name}_X_validation.parquet",
        index=True
    )

    X_test_processed_df.to_parquet(
        SPLIT_DIR
        / f"{dataset_name}_X_test.parquet",
        index=True
    )

    y_train.to_frame(
        name=target_column
    ).to_parquet(
        SPLIT_DIR
        / f"{dataset_name}_y_train.parquet",
        index=True
    )

    y_valid.to_frame(
        name=target_column
    ).to_parquet(
        SPLIT_DIR
        / f"{dataset_name}_y_validation.parquet",
        index=True
    )

    y_test.to_frame(
        name=target_column
    ).to_parquet(
        SPLIT_DIR
        / f"{dataset_name}_y_test.parquet",
        index=True
    )

    # --------------------------------------------------------------
    # Save preprocessor
    # --------------------------------------------------------------

    preprocessor_file = (
        PREPROCESSOR_DIR
        / f"{dataset_name}_preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        preprocessor_file
    )

    # --------------------------------------------------------------
    # Save feature names
    # --------------------------------------------------------------

    feature_names_file = (
        PREPROCESSOR_DIR
        / f"{dataset_name}_feature_names.txt"
    )

    with open(
        feature_names_file,
        "w",
        encoding="utf-8"
    ) as f:

        for feature in transformed_feature_names:

            f.write(
                f"{feature}\n"
            )

    # --------------------------------------------------------------
    # Final verification
    # --------------------------------------------------------------

    print_section(
        f"{dataset_name.upper()} FINAL VERIFICATION"
    )

    print(
        f"Original rows       : {len(df):,}"
    )

    print(
        f"Training rows       : {len(X_train):,}"
    )

    print(
        f"Validation rows     : {len(X_valid):,}"
    )

    print(
        f"Test rows           : {len(X_test):,}"
    )

    print(
        f"Raw predictor count : {len(X_model.columns):,}"
    )

    print(
        f"Processed features  : "
        f"{len(transformed_feature_names):,}"
    )

    print(
        f"Target              : "
        f"{target_column}"
    )

    print(
        f"Preprocessor saved  : "
        f"{preprocessor_file}"
    )

    print(
        "\nDataset processing complete."
    )

    # --------------------------------------------------------------
    # Return summary
    # --------------------------------------------------------------

    return {
        "dataset": dataset_name,
        "rows": len(df),
        "train_rows": len(X_train),
        "validation_rows": len(X_valid),
        "test_rows": len(X_test),
        "raw_features": len(X_model.columns),
        "processed_features":
            len(transformed_feature_names),
        "categorical_features":
            len(categorical_features),
        "numerical_features":
            len(numerical_features),
        "target": target_column
    }


# ==================================================================
# 6. RUN ALL DATASETS
# ==================================================================

print_section(
    "PHASE 3.1 - DATA PREPARATION START"
)

summary_records = []

for dataset_name, config in DATASETS.items():

    result = process_dataset(
        dataset_name,
        config
    )

    summary_records.append(
        result
    )


# ==================================================================
# 7. SUMMARY
# ==================================================================

print_section(
    "PHASE 3.1 FINAL SUMMARY"
)

summary_df = pd.DataFrame(
    summary_records
)

print(
    summary_df.to_string(
        index=False
    )
)

summary_df.to_csv(
    REPORT_DIR
    / "phase3_1_split_summary.csv",
    index=False
)


# ==================================================================
# 8. COMPLETION
# ==================================================================

print_section(
    "PHASE 3.1 COMPLETE"
)

print(
    """
Train / validation / test splitting completed.

Preprocessing completed.

Important:
    - loan_id was excluded from ML predictors.
    - Splits were stratified.
    - No train/validation/test overlap exists.
    - Preprocessors were fitted ONLY on training data.
    - Validation and test data were transformed using
      the training-fitted preprocessors.
    - No ML model was trained.
    - No SMOTE was applied.

Ready for Phase 3.2:
    Baseline Model Training.
"""
)