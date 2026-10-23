
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models" / "phase3" / "baseline"
SPLIT_DIR = ROOT / "data" / "phase3" / "splits"

OUTPUT_DIR = (
    ROOT / "phase3" / "reports" / "explainability"
    / "calibration_assessment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TASK CONFIGURATION
# ============================================================

TASKS = {
    "model_a": {
        "name": "Model A - Origination Risk",
        "model_file": "model_a_gradient_boosting.joblib",
        "x_file": "model_a_X_validation.parquet",
        "y_file": "model_a_y_validation.parquet",
        "threshold": 0.1983,
    },
    "model_b_3m": {
        "name": "Model B - 3-Month Early Warning",
        "model_file": "model_b_3m_gradient_boosting.joblib",
        "x_file": "model_b_3m_X_validation.parquet",
        "y_file": "model_b_3m_y_validation.parquet",
        "threshold": 0.2338,
    },
    "model_b_6m": {
        "name": "Model B - 6-Month Early Warning",
        "model_file": "model_b_6m_gradient_boosting.joblib",
        "x_file": "model_b_6m_X_validation.parquet",
        "y_file": "model_b_6m_y_validation.parquet",
        "threshold": 0.3646,
    },
}


# ============================================================
# LOAD TARGET
# ============================================================

def load_target(path):
    data = pd.read_parquet(path)

    if data.shape[1] != 1:
        raise ValueError(
            f"Expected one target column in {path}; "
            f"found {data.shape[1]}."
        )

    return data.iloc[:, 0].to_numpy()


# ============================================================
# RISK BAND REPORT
# ============================================================

def create_risk_band_report(y_true, scores):
    # These are illustrative bands, not validated lending rules.
    band_labels = ["Low score", "Medium score", "High score"]

    bands = pd.cut(
        scores,
        bins=[-np.inf, 0.20, 0.50, np.inf],
        labels=band_labels,
        right=False,
    )

    data = pd.DataFrame({
        "score": scores,
        "actual_target": y_true,
        "risk_band": bands,
    })

    report = (
        data.groupby("risk_band", observed=False)
        .agg(
            record_count=("actual_target", "size"),
            actual_positive_count=("actual_target", "sum"),
            mean_predicted_score=("score", "mean"),
            observed_positive_rate=("actual_target", "mean"),
        )
        .reset_index()
    )

    report["observed_positive_rate_pct"] = (
        report["observed_positive_rate"] * 100
    )
    report["mean_predicted_score_pct"] = (
        report["mean_predicted_score"] * 100
    )

    return report


# ============================================================
# CALIBRATION REPORT
# ============================================================

def create_calibration_report(y_true, scores, task_key):
    # Equal-width probability bins across [0, 1].
    observed_rates, mean_scores = calibration_curve(
        y_true,
        scores,
        n_bins=10,
        strategy="uniform",
    )

    calibration_df = pd.DataFrame({
        "mean_predicted_score": mean_scores,
        "observed_positive_rate": observed_rates,
    })

    calibration_df["absolute_gap"] = np.abs(
        calibration_df["mean_predicted_score"]
        - calibration_df["observed_positive_rate"]
    )

    calibration_df.to_csv(
        OUTPUT_DIR / f"{task_key}_calibration_bins.csv",
        index=False,
    )

    # Calibration plot.
    plt.figure(figsize=(7, 6))
    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )
    plt.plot(
        mean_scores,
        observed_rates,
        marker="o",
        label="Model",
    )

    plt.xlabel("Mean predicted score")
    plt.ylabel("Observed positive rate")
    plt.title(f"Calibration Curve - {task_key}")
    plt.legend()
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"{task_key}_calibration_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return calibration_df


# ============================================================
# PROCESS ONE MODEL
# ============================================================

def assess_task(task_key, config):
    print("\n" + "=" * 70)
    print(config["name"])
    print("=" * 70)

    model_path = MODEL_DIR / config["model_file"]
    x_path = SPLIT_DIR / config["x_file"]
    y_path = SPLIT_DIR / config["y_file"]

    for path in [model_path, x_path, y_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    model = joblib.load(model_path)
    X_val = pd.read_parquet(x_path)
    y_val = load_target(y_path)

    if len(X_val) != len(y_val):
        raise ValueError("Validation feature/target size mismatch.")

    scores = model.predict_proba(X_val)[:, 1]

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction scores detected.")

    scores = np.clip(scores, 0.0, 1.0)

    # Overall validation calibration metrics.
    brier = brier_score_loss(y_val, scores)
    clipped_scores = np.clip(scores, 1e-15, 1 - 1e-15)
    cross_entropy = log_loss(y_val, clipped_scores, labels=[0, 1])

    print(f"Validation rows: {len(y_val):,}")
    print(f"Observed positive rate: {np.mean(y_val):.4f}")
    print(f"Mean predicted score: {np.mean(scores):.4f}")
    print(f"Brier score (lower is better): {brier:.4f}")
    print(f"Log loss (lower is better): {cross_entropy:.4f}")

    # Save record-level validation scores.
    record_df = pd.DataFrame({
        "validation_row": np.arange(len(y_val)),
        "actual_target": y_val,
        "predicted_score": scores,
        "predicted_class_at_validation_threshold": (
            scores >= config["threshold"]
        ).astype(int),
    })

    record_df.to_csv(
        OUTPUT_DIR / f"{task_key}_validation_scores.csv",
        index=False,
    )

    # Risk bands.
    band_report = create_risk_band_report(y_val, scores)
    band_report.to_csv(
        OUTPUT_DIR / f"{task_key}_risk_band_validation.csv",
        index=False,
    )

    print("\nRisk band validation:")
    print(
        band_report[
            [
                "risk_band",
                "record_count",
                "actual_positive_count",
                "mean_predicted_score_pct",
                "observed_positive_rate_pct",
            ]
        ].round(2).to_string(index=False)
    )

    # Calibration bins and plot.
    calibration_df = create_calibration_report(
        y_val, scores, task_key
    )

    print("\nCalibration bins:")
    print(calibration_df.round(4).to_string(index=False))

    return {
        "task": task_key,
        "validation_rows": len(y_val),
        "observed_positive_rate": np.mean(y_val),
        "mean_predicted_score": np.mean(scores),
        "brier_score": brier,
        "log_loss": cross_entropy,
        "validation_threshold": config["threshold"],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    summaries = []

    for task_key, config in TASKS.items():
        summaries.append(assess_task(task_key, config))

    summary_df = pd.DataFrame(summaries)

    summary_path = OUTPUT_DIR / "calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("PHASE 3.4.3 COMPLETED")
    print("=" * 70)
    print(summary_df.round(4).to_string(index=False))
    print(f"\nReports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
