
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models" / "phase3" / "baseline"
PREPROCESSOR_DIR = ROOT / "data" / "phase3" / "preprocessors"
OUTPUT_DIR = ROOT / "models" / "phase3" / "packaged"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = {
    "model_a": {
        "model_file": "model_a_gradient_boosting.joblib",
        "threshold": 0.1983,
        "purpose": "Origination risk",
    },
    "model_b_3m": {
        "model_file": "model_b_3m_gradient_boosting.joblib",
        "threshold": 0.2338,
        "purpose": "3-month early warning",
    },
    "model_b_6m": {
        "model_file": "model_b_6m_gradient_boosting.joblib",
        "threshold": 0.3646,
        "purpose": "6-month early warning",
    },
}


def find_preprocessor(task_key):
    # Match the task name and a preprocessing-related filename.
    files = list(PREPROCESSOR_DIR.glob("*.joblib"))
    matches = [
        path for path in files
        if task_key in path.stem.lower()
        and any(
            word in path.stem.lower()
            for word in ("preprocessor", "preprocess", "transformer")
        )
    ]

    if len(matches) != 1:
        available = "\n".join(str(p.name) for p in files)
        raise FileNotFoundError(
            f"\nExpected exactly one preprocessor for {task_key}; "
            f"found {len(matches)}.\n"
            f"Preprocessor directory: {PREPROCESSOR_DIR}\n"
            f"Available .joblib files:\n{available}\n"
            "Update find_preprocessor() to match your actual filename."
        )

    return matches[0]


def package_task(task_key, config):
    model_path = MODEL_DIR / config["model_file"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    preprocessor_path = find_preprocessor(task_key)

    classifier = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    if not hasattr(preprocessor, "transform"):
        raise TypeError(
            f"Artifact does not support transform(): {preprocessor_path}"
        )

    if not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError(
            "Preprocessor does not expose feature_names_in_. "
            "Confirm it was fitted on the original raw input columns."
        )

    required_columns = list(preprocessor.feature_names_in_)

    # Check that preprocessing produces the number of features
    # expected by the already-trained classifier.
    if hasattr(preprocessor, "n_features_in_"):
        if preprocessor.n_features_in_ != len(required_columns):
            raise ValueError("Preprocessor input feature count mismatch.")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    bundle = {
        "task": task_key,
        "purpose": config["purpose"],
        "pipeline": pipeline,
        "required_columns": required_columns,
        "threshold": config["threshold"],
        "model_type": "GradientBoostingClassifier",
        "dataset_type": "synthetic credit-risk data",
        "threshold_source": "validation set",
        "score_note": (
            "Predicted score is not guaranteed to be calibrated."
        ),
    }

    output_path = OUTPUT_DIR / f"{task_key}_bundle.joblib"
    joblib.dump(bundle, output_path)

    print(f"\nTask: {task_key}")
    print(f"Model: {model_path.name}")
    print(f"Preprocessor: {preprocessor_path.name}")
    print(f"Required raw input columns: {len(required_columns)}")
    print(f"Saved bundle: {output_path}")


def main():
    for task_key, config in TASKS.items():
        package_task(task_key, config)

    print("\nAll model bundles packaged successfully.")


if __name__ == "__main__":
    main()
