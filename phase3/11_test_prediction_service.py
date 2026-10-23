from pathlib import Path

import pandas as pd

from prediction_service import predict_risk


ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "origination": (
        ROOT / "data/processed/model_a_origination_dataset.parquet",
    ),
    "early_warning_3m": (
        ROOT / "data/processed/model_b_final_early_warning_3m.parquet",
    ),
    "early_warning_6m": (
        ROOT / "data/processed/model_b_final_early_warning_6m.parquet",
    ),
}


def main():
    print("=" * 65)
    print("PHASE 3.5 - END-TO-END PREDICTION SERVICE TEST")
    print("=" * 65)

    for task, (dataset_path,) in DATASETS.items():
        print(f"\nTesting: {task}")

        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}")
            continue

        # Load the original processed dataset
        df = pd.read_parquet(dataset_path)

        # Load the corresponding packaged model bundle
        bundle_names = {
            "origination": "model_a_bundle.joblib",
            "early_warning_3m": "model_b_3m_bundle.joblib",
            "early_warning_6m": "model_b_6m_bundle.joblib",
        }

        import joblib

        bundle_path = (
            ROOT / "models/phase3/packaged" / bundle_names[task]
        )
        bundle = joblib.load(bundle_path)

        required_columns = bundle["required_columns"]

        # Ensure every required predictor exists in the dataset
        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]

        if missing_columns:
            print("Missing required dataset columns:", missing_columns)
            continue

        # Use one raw example; exclude target and unrelated columns
        raw_input = df.iloc[0][required_columns].to_dict()

        # Use the same prediction interface intended for the backend
        result = predict_risk(task, raw_input)

        score = result["predicted_score"]
        threshold = result["validation_threshold"]

        print(f"Input fields: {len(raw_input)}")
        print(f"Predicted score: {score:.6f}")
        print(f"Validation threshold: {threshold:.4f}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Interpretation: {result['interpretation']}")

        # Basic output validation
        assert 0 <= score <= 1, "Score is outside [0, 1]"
        assert 0 <= threshold <= 1, "Threshold is outside [0, 1]"
        assert result["predicted_class"] == int(score >= threshold)

        print("PASS: Raw input -> preprocessing -> prediction")

    print("\nEnd-to-end smoke test finished.")


if __name__ == "__main__":
    main()
