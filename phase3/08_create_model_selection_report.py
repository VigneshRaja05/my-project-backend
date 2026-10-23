
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "phase3" / "reports" / "model_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Final test metrics from Phase 3.3.
# Thresholds were selected using validation data.
models = [
    {
        "task": "model_a",
        "model": "Gradient Boosting",
        "test_rows": 15000,
        "test_roc_auc": 0.7881,
        "test_pr_auc": 0.4266,
        "validation_threshold": 0.1983,
        "purpose": "Origination risk",
    },
    {
        "task": "model_b_3m",
        "model": "Gradient Boosting",
        "test_rows": 7804,
        "test_roc_auc": 0.8160,
        "test_pr_auc": 0.6625,
        "validation_threshold": 0.2338,
        "purpose": "Early warning using first 3 months",
    },
    {
        "task": "model_b_6m",
        "model": "Gradient Boosting",
        "test_rows": 7667,
        "test_roc_auc": 0.8216,
        "test_pr_auc": 0.6617,
        "validation_threshold": 0.3646,
        "purpose": "Early warning using first 6 months",
    },
]

df = pd.DataFrame(models)
df["selection_basis"] = "Highest validation PR-AUC among baseline candidates"
df["dataset_note"] = "Synthetic credit-risk dataset"
df["threshold_source"] = "Validation set; not optimized on test set"

csv_path = OUTPUT_DIR / "selected_model_summary.csv"
df.to_csv(csv_path, index=False)

report_path = OUTPUT_DIR / "selected_model_summary.txt"

with report_path.open("w", encoding="utf-8") as f:
    f.write("MSME LOAN RISK PROJECT - MODEL SELECTION\n")
    f.write("=" * 60 + "\n\n")
    f.write("Selected models: Gradient Boosting for all three tasks.\n")
    f.write("Selection criterion: validation PR-AUC.\n")
    f.write("Final test results are reported without further tuning.\n\n")
    f.write(df.to_string(index=False))
    f.write(
        "\n\nLimitations:\n"
        "- Dataset is synthetic, not verified Indian MSME loan data.\n"
        "- Predicted scores are not automatically calibrated probabilities.\n"
        "- Thresholds are prototype operating points, not lending policies.\n"
    )

print(df.to_string(index=False))
print(f"\nSaved CSV: {csv_path}")
print(f"Saved report: {report_path}")
