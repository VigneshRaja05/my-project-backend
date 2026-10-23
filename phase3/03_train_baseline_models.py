
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
    precision_recall_curve,
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = ROOT / "data" / "phase3" / "splits"
MODEL_DIR = ROOT / "models" / "phase3" / "baseline"
REPORT_DIR = ROOT / "phase3" / "reports" / "baseline"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

DATASETS = {
    "model_a": {
        "target": "defaulted",
    },
    "model_b_3m": {
        "target": "early_warning_target_3m",
    },
    "model_b_6m": {
        "target": "early_warning_target_6m",
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_split(dataset_name, split_name, data_type):
    """
    Expected examples:
        model_a_X_train.parquet
        model_a_y_train.parquet
        model_b_3m_X_validation.parquet
        model_b_3m_y_validation.parquet
    """
    filename = (
        f"{dataset_name}_{data_type}_{split_name}.parquet"
    )
    path = SPLIT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n"
            "Check the filenames in data/phase3/splits."
        )

    return pd.read_parquet(path)


def load_target(dataset_name, split_name, target_name):
    y_df = load_split(dataset_name, split_name, "y")

    if target_name in y_df.columns:
        y = y_df[target_name]
    elif y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
    else:
        raise ValueError(
            f"Could not identify target '{target_name}' in "
            f"{dataset_name}_{split_name}: {list(y_df.columns)}"
        )

    return pd.to_numeric(y, errors="raise").astype(int)


def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "specificity": specificity,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
    }


def find_best_f1_threshold(y_true, probabilities):
    """
    Select a threshold using validation data only.
    This threshold must not be selected using test data.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index])


def save_feature_importance(model, feature_names, output_path):
    """
    Save model-based feature importance where supported.
    Logistic Regression uses absolute coefficients.
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        method = "tree_feature_importance"

    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).mean(axis=0)
        method = "absolute_logistic_coefficient"

    else:
        return

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "importance_method": method,
    })

    result.sort_values(
        "importance", ascending=False
    ).to_csv(output_path, index=False)


# ============================================================
# BASELINE MODELS
# ============================================================

def create_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    all_results = []

    print("=" * 75)
    print("PHASE 3.2 — BASELINE MODEL TRAINING")
    print("=" * 75)
    print(f"Split directory : {SPLIT_DIR}")
    print(f"Model directory : {MODEL_DIR}")
    print(f"Report directory: {REPORT_DIR}")

    for dataset_name, config in DATASETS.items():
        target_name = config["target"]

        print("\n" + "=" * 75)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 75)

        # Load preprocessed datasets
        X_train = load_split(
            dataset_name, "train", "X"
        )
        X_val = load_split(
            dataset_name, "validation", "X"
        )

        y_train = load_target(
            dataset_name, "train", target_name
        )
        y_val = load_target(
            dataset_name, "validation", target_name
        )

        # Ensure the train and validation schemas match
        if list(X_train.columns) != list(X_val.columns):
            raise ValueError(
                f"Feature mismatch for {dataset_name}."
            )

        if len(X_train) != len(y_train):
            raise ValueError("Training X/y row counts differ.")

        if len(X_val) != len(y_val):
            raise ValueError("Validation X/y row counts differ.")

        if y_train.nunique() != 2 or y_val.nunique() != 2:
            raise ValueError(
                f"{dataset_name} must contain both target classes "
                "in train and validation data."
            )

        print(f"Training shape   : {X_train.shape}")
        print(f"Validation shape : {X_val.shape}")
        print(f"Training classes : {y_train.value_counts().to_dict()}")
        print(f"Validation classes: {y_val.value_counts().to_dict()}")

        models = create_models()

        for model_name, model in models.items():
            print("\n" + "-" * 65)
            print(f"Training: {model_name}")
            print("-" * 65)

            # Fit only on training data
            model.fit(X_train, y_train)

            # Validation probabilities
            probabilities = model.predict_proba(X_val)[:, 1]

            # Standard metrics at threshold 0.50
            metrics_05 = calculate_metrics(
                y_val, probabilities, threshold=0.5
            )

            # Choose an alternative threshold on validation only
            best_threshold = find_best_f1_threshold(
                y_val, probabilities
            )

            metrics_tuned = calculate_metrics(
                y_val,
                probabilities,
                threshold=best_threshold,
            )

            result = {
                "dataset": dataset_name,
                "model": model_name,
                "target": target_name,
                "train_rows": len(X_train),
                "validation_rows": len(X_val),
                "features": X_train.shape[1],
                "best_f1_threshold": best_threshold,
            }

            # Preserve both standard and tuned-threshold results
            for metric_name, value in metrics_05.items():
                result[f"default_{metric_name}"] = value

            for metric_name, value in metrics_tuned.items():
                result[f"tuned_{metric_name}"] = value

            all_results.append(result)

            # Save the fitted estimator
            model_path = (
                MODEL_DIR
                / f"{dataset_name}_{model_name}.joblib"
            )
            joblib.dump(model, model_path)

            # Save validation predictions
            prediction_path = (
                REPORT_DIR
                / f"{dataset_name}_{model_name}_validation_predictions.csv"
            )

            pd.DataFrame({
                "actual": y_val.to_numpy(),
                "predicted_probability": probabilities,
                "predicted_class_at_0_50": (
                    probabilities >= 0.5
                ).astype(int),
                "predicted_class_at_tuned_threshold": (
                    probabilities >= best_threshold
                ).astype(int),
            }).to_csv(prediction_path, index=False)

            # Save feature importance for interpretation
            importance_path = (
                REPORT_DIR
                / f"{dataset_name}_{model_name}_feature_importance.csv"
            )

            save_feature_importance(
                model,
                X_train.columns,
                importance_path,
            )

            print(f"ROC-AUC          : {metrics_05['roc_auc']:.4f}")
            print(f"PR-AUC           : {metrics_05['pr_auc']:.4f}")
            print(f"Precision @ 0.50 : {metrics_05['precision']:.4f}")
            print(f"Recall @ 0.50    : {metrics_05['recall']:.4f}")
            print(f"F1 @ 0.50        : {metrics_05['f1']:.4f}")
            print(f"Brier score      : {metrics_05['brier_score']:.4f}")
            print(f"Best F1 threshold: {best_threshold:.4f}")
            print(f"Tuned F1         : {metrics_tuned['f1']:.4f}")
            print(f"Saved model      : {model_path}")

    # Save combined experiment results
    results_df = pd.DataFrame(all_results)

    results_path = REPORT_DIR / "baseline_model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    # Rank separately within each prediction task
    ranking = results_df.sort_values(
        ["dataset", "default_pr_auc", "default_roc_auc"],
        ascending=[True, False, False],
    )

    ranking_path = REPORT_DIR / "baseline_model_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    print("\n" + "=" * 75)
    print("PHASE 3.2 — VALIDATION COMPARISON")
    print("=" * 75)

    display_columns = [
        "dataset",
        "model",
        "default_roc_auc",
        "default_pr_auc",
        "default_precision",
        "default_recall",
        "default_f1",
        "default_brier_score",
        "best_f1_threshold",
        "tuned_f1",
        "tuned_recall",
    ]

    print(
        ranking[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved comparison:", results_path)
    print("Saved ranking   :", ranking_path)

    print("\n" + "=" * 75)
    print("PHASE 3.2 BASELINE TRAINING COMPLETE")
    print("The test sets were not loaded or evaluated.")
    print("=" * 75)


if __name__ == "__main__":
    main()
