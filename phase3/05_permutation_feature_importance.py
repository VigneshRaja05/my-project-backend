
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models" / "phase3" / "baseline"
SPLIT_DIR = ROOT / "data" / "phase3" / "splits"
OUTPUT_DIR = ROOT / "phase3" / "reports" / "explainability"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

TASKS = {
    "model_a": {
        "name": "Model A - Origination Risk",
        "model_file": "model_a_gradient_boosting.joblib",
        "x_file": "model_a_X_validation.parquet",
        "y_file": "model_a_y_validation.parquet",
        "target": "defaulted",
    },
    "model_b_3m": {
        "name": "Model B - 3-Month Early Warning",
        "model_file": "model_b_3m_gradient_boosting.joblib",
        "x_file": "model_b_3m_X_validation.parquet",
        "y_file": "model_b_3m_y_validation.parquet",
        "target": "early_warning_target_3m",
    },
    "model_b_6m": {
        "name": "Model B - 6-Month Early Warning",
        "model_file": "model_b_6m_gradient_boosting.joblib",
        "x_file": "model_b_6m_X_validation.parquet",
        "y_file": "model_b_6m_y_validation.parquet",
        "target": "early_warning_target_6m",
    },
}


# ============================================================
# HELPER: LOAD TARGET COLUMN
# ============================================================

def load_target(path):
    data = pd.read_parquet(path)

    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError(
                f"Expected one target column in {path}, "
                f"but found {data.shape[1]} columns."
            )
        return data.iloc[:, 0].to_numpy()

    return np.asarray(data).reshape(-1)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def analyze_task(task_key, config):
    print("\n" + "=" * 70)
    print(config["name"])
    print("=" * 70)

    model_path = MODEL_DIR / config["model_file"]
    x_path = SPLIT_DIR / config["x_file"]
    y_path = SPLIT_DIR / config["y_file"]

    for path in [model_path, x_path, y_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    model = joblib.load(model_path)
    X_val = pd.read_parquet(x_path)
    y_val = load_target(y_path)

    if len(X_val) != len(y_val):
        raise ValueError(
            f"Feature/target row mismatch for {task_key}: "
            f"{len(X_val)} vs {len(y_val)}"
        )

    print(f"Validation rows: {len(X_val):,}")
    print(f"Encoded features: {X_val.shape[1]}")
    print(f"Positive cases: {int(np.sum(y_val == 1)):,}")

    baseline_probabilities = model.predict_proba(X_val)[:, 1]
    baseline_pr_auc = average_precision_score(
        y_val, baseline_probabilities
    )

    print(f"Baseline validation PR-AUC: {baseline_pr_auc:.4f}")
    print("Calculating permutation importance...")

    result = permutation_importance(
        estimator=model,
        X=X_val,
        y=y_val,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False
    ).reset_index(drop=True)

    importance_df.insert(
        0, "rank", np.arange(1, len(importance_df) + 1)
    )

    output_file = OUTPUT_DIR / (
        f"{task_key}_permutation_importance.csv"
    )
    importance_df.to_csv(output_file, index=False)

    top_features = importance_df.head(15)

    print("\nTop 15 features:")
    print(
        top_features[
            ["rank", "feature", "importance_mean", "importance_std"]
        ].to_string(index=False)
    )

    print(f"\nSaved: {output_file}")

    return importance_df


# ============================================================
# MAIN
# ============================================================

def main():
    all_results = []

    for task_key, config in TASKS.items():
        importance_df = analyze_task(task_key, config)

        importance_df.insert(
            0, "task", task_key
        )
        all_results.append(importance_df)

    combined = pd.concat(all_results, ignore_index=True)

    combined_file = OUTPUT_DIR / "all_tasks_feature_importance.csv"
    combined.to_csv(combined_file, index=False)

    print("\n" + "=" * 70)
    print("PHASE 3.4 - FEATURE IMPORTANCE COMPLETED")
    print("=" * 70)
    print(f"Combined report: {combined_file}")
    print(f"Individual reports: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
