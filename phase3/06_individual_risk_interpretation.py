
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models" / "phase3" / "baseline"
SPLIT_DIR = ROOT / "data" / "phase3" / "splits"
IMPORTANCE_DIR = ROOT / "phase3" / "reports" / "explainability"
OUTPUT_DIR = IMPORTANCE_DIR / "individual_risk_reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TASKS = {
    "model_a": {
        "name": "Origination Risk",
        "model": "model_a_gradient_boosting.joblib",
        "x": "model_a_X_validation.parquet",
        "y": "model_a_y_validation.parquet",
        "importance": "model_a_permutation_importance.csv",
    },
    "model_b_3m": {
        "name": "3-Month Early Warning",
        "model": "model_b_3m_gradient_boosting.joblib",
        "x": "model_b_3m_X_validation.parquet",
        "y": "model_b_3m_y_validation.parquet",
        "importance": "model_b_3m_permutation_importance.csv",
    },
    "model_b_6m": {
        "name": "6-Month Early Warning",
        "model": "model_b_6m_gradient_boosting.joblib",
        "x": "model_b_6m_X_validation.parquet",
        "y": "model_b_6m_y_validation.parquet",
        "importance": "model_b_6m_permutation_importance.csv",
    },
}


def load_target(path):
    data = pd.read_parquet(path)

    if data.shape[1] != 1:
        raise ValueError(
            f"Expected exactly one target column in {path}"
        )

    return data.iloc[:, 0].to_numpy()


def assign_score_band(score):
    # Illustrative bands only; not calibrated risk categories.
    if score < 0.20:
        return "Low score"
    elif score < 0.50:
        return "Medium score"
    else:
        return "High score"


def generate_report(task_key, config):
    print("\n" + "=" * 65)
    print(config["name"])
    print("=" * 65)

    model = joblib.load(MODEL_DIR / config["model"])
    X_val = pd.read_parquet(SPLIT_DIR / config["x"])
    y_val = load_target(SPLIT_DIR / config["y"])

    importance = pd.read_csv(
        IMPORTANCE_DIR / config["importance"]
    )

    if len(X_val) != len(y_val):
        raise ValueError("Validation feature/target size mismatch.")

    # Use only the validation set in this script.
    scores = model.predict_proba(X_val)[:, 1]

    report = pd.DataFrame({
        "validation_row": np.arange(len(X_val)),
        "actual_target": y_val,
        "predicted_score": scores,
    })

    report["illustrative_score_band"] = (
        report["predicted_score"].apply(assign_score_band)
    )

    # Attach the values of the five globally most important features.
    top_features = importance.head(5)["feature"].tolist()

    for rank, feature in enumerate(top_features, start=1):
        if feature not in X_val.columns:
            print(f"Warning: feature not found: {feature}")
            continue

        report[f"top_feature_{rank}"] = feature
        report[f"feature_{rank}_value"] = X_val[feature].to_numpy()

    report = report.sort_values(
        "predicted_score", ascending=False
    ).reset_index(drop=True)

    # Save scores for every validation record.
    all_path = OUTPUT_DIR / f"{task_key}_all_validation_scores.csv"
    report.to_csv(all_path, index=False)

    # Save the 100 highest-scoring and 100 lowest-scoring examples.
    high_path = OUTPUT_DIR / f"{task_key}_highest_scores.csv"
    low_path = OUTPUT_DIR / f"{task_key}_lowest_scores.csv"

    report.head(100).to_csv(high_path, index=False)
    report.tail(100).sort_values(
        "predicted_score"
    ).to_csv(low_path, index=False)

    print(f"Validation records: {len(report):,}")
    print(f"Actual positive cases: {int(np.sum(y_val == 1)):,}")
    print(f"Mean predicted score: {scores.mean():.4f}")

    print("\nIllustrative score-band counts:")
    print(
        report["illustrative_score_band"]
        .value_counts()
        .reindex(
            ["Low score", "Medium score", "High score"],
            fill_value=0
        )
        .to_string()
    )

    print("\nHighest-scoring examples:")
    print(
        report.head(5)[
            [
                "validation_row",
                "actual_target",
                "predicted_score",
                "illustrative_score_band",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {all_path}")
    print(f"Saved: {high_path}")
    print(f"Saved: {low_path}")


def main():
    for task_key, config in TASKS.items():
        generate_report(task_key, config)

    print("\n" + "=" * 65)
    print("INDIVIDUAL RISK REPORTS GENERATED")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
