
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = ROOT / "data" / "phase3" / "splits"
MODEL_DIR = ROOT / "models" / "phase3" / "baseline"
REPORT_DIR = ROOT / "phase3" / "reports" / "baseline"

COMPARISON_FILE = REPORT_DIR / "baseline_model_comparison.csv"

TARGETS = {
    "model_a": "defaulted",
    "model_b_3m": "early_warning_target_3m",
    "model_b_6m": "early_warning_target_6m",
}


def load_parquet(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)


def load_target(dataset, target_name):
    path = SPLIT_DIR / f"{dataset}_y_test.parquet"
    df = load_parquet(path)

    if target_name in df.columns:
        y = df[target_name]
    elif df.shape[1] == 1:
        y = df.iloc[:, 0]
    else:
        raise ValueError(
            f"Cannot identify target {target_name} in {path}"
        )

    return pd.to_numeric(y, errors="raise").astype(int)


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

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
        "specificity": (
            tn / (tn + fp) if (tn + fp) else 0.0
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
    }


def main():
    print("=" * 75)
    print("PHASE 3.3 — FINAL TEST EVALUATION")
    print("=" * 75)

    comparison = pd.read_csv(COMPARISON_FILE)

    required_columns = {
        "dataset",
        "model",
        "default_pr_auc",
        "best_f1_threshold",
    }

    if not required_columns.issubset(comparison.columns):
        raise ValueError(
            f"Comparison file needs columns: {required_columns}"
        )

    # Select winners using validation PR-AUC only.
    # Test data is not used for model selection.
    selected = (
        comparison.sort_values(
            ["dataset", "default_pr_auc"],
            ascending=[True, False],
        )
        .groupby("dataset", as_index=False)
        .first()
    )

    all_metrics = []

    for _, row in selected.iterrows():
        dataset = row["dataset"]
        model_name = row["model"]
        target_name = TARGETS[dataset]
        threshold = float(row["best_f1_threshold"])

        print("\n" + "=" * 75)
        print(f"DATASET: {dataset.upper()}")
        print(f"Selected model: {model_name}")
        print(f"Validation PR-AUC: {row['default_pr_auc']:.4f}")
        print(f"Validation-selected threshold: {threshold:.4f}")
        print("=" * 75)

        model_path = MODEL_DIR / f"{dataset}_{model_name}.joblib"
        model = joblib.load(model_path)

        X_test = load_parquet(
            SPLIT_DIR / f"{dataset}_X_test.parquet"
        )
        y_test = load_target(dataset, target_name)

        if len(X_test) != len(y_test):
            raise ValueError(
                f"Test X/y length mismatch for {dataset}"
            )

        probabilities = model.predict_proba(X_test)[:, 1]

        # Report both thresholds. Do not tune using test labels.
        for threshold_name, cutoff in [
            ("default_0_50", 0.5),
            ("validation_tuned", threshold),
        ]:
            metrics = evaluate(y_test, probabilities, cutoff)
            metrics.update({
                "dataset": dataset,
                "model": model_name,
                "threshold_type": threshold_name,
                "test_rows": len(y_test),
                "validation_pr_auc": float(row["default_pr_auc"]),
            })
            all_metrics.append(metrics)

            print(f"\n--- {threshold_name} ---")
            for key in [
                "roc_auc",
                "pr_auc",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "specificity",
                "brier_score",
                "true_negatives",
                "false_positives",
                "false_negatives",
                "true_positives",
            ]:
                print(f"{key:22s}: {metrics[key]:.4f}"
                      if isinstance(metrics[key], float)
                      else f"{key:22s}: {metrics[key]}")

        # Save test probabilities for subsequent analysis.
        prediction_path = (
            REPORT_DIR / f"{dataset}_{model_name}_test_predictions.csv"
        )

        pd.DataFrame({
            "actual": y_test.to_numpy(),
            "predicted_probability": probabilities,
            "predicted_at_0_50": (probabilities >= 0.5).astype(int),
            "predicted_at_validation_threshold": (
                probabilities >= threshold
            ).astype(int),
        }).to_csv(prediction_path, index=False)

    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = REPORT_DIR / "final_test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    selected_path = REPORT_DIR / "selected_models.csv"
    selected.to_csv(selected_path, index=False)

    print("\n" + "=" * 75)
    print("PHASE 3.3 COMPLETE")
    print("=" * 75)
    print(metrics_df[
        [
            "dataset",
            "model",
            "threshold_type",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "brier_score",
        ]
    ].round(4).to_string(index=False))

    print(f"\nFinal metrics saved: {metrics_path}")
    print(f"Selected models saved: {selected_path}")
    print("Test data was used only for final evaluation.")


if __name__ == "__main__":
    main()
