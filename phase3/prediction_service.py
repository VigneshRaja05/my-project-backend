from collections.abc import Mapping
from numbers import Real
from pathlib import Path
import math

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = ROOT / "models" / "phase3" / "packaged"

BUNDLE_FILES = {
    "origination": "model_a_bundle.joblib",
    "early_warning_3m": "model_b_3m_bundle.joblib",
    "early_warning_6m": "model_b_6m_bundle.joblib",
}

PURPOSES = {
    "origination": "Origination risk",
    "early_warning_3m": "Early warning using first 3 months",
    "early_warning_6m": "Early warning using first 6 months",
}

# Load each bundle only once per running Python process.
_BUNDLE_CACHE = {}


class PredictionInputError(ValueError):
    """Raised when prediction input is invalid."""


def _load_bundle(task):
    if task not in BUNDLE_FILES:
        supported = ", ".join(BUNDLE_FILES)
        raise PredictionInputError(
            f"Unknown task '{task}'. Supported tasks: {supported}"
        )

    if task not in _BUNDLE_CACHE:
        path = BUNDLE_DIR / BUNDLE_FILES[task]

        if not path.exists():
            raise FileNotFoundError(
                f"Packaged model not found: {path}"
            )

        _BUNDLE_CACHE[task] = joblib.load(path)

    return _BUNDLE_CACHE[task]


def _is_missing(value):
    if value is None:
        return True

    try:
        result = pd.isna(value)
        return bool(result)
    except (TypeError, ValueError):
        return False


def _get_column_groups(preprocessor):
    """Identify raw numeric and categorical columns from the fitted transformer."""
    numeric_columns = set()
    categorical_columns = set()

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue

        if isinstance(columns, str):
            columns = [columns]
        elif not isinstance(columns, (list, tuple, set)):
            try:
                columns = list(columns)
            except TypeError:
                continue

        params = {}
        if hasattr(transformer, "get_params"):
            params = transformer.get_params(deep=True)

        contains_onehot = any(
            isinstance(value, OneHotEncoder)
            for value in params.values()
        )

        if "categor" in name.lower() or contains_onehot:
            categorical_columns.update(columns)
        else:
            numeric_columns.update(columns)

    return numeric_columns, categorical_columns


def _validate_input(task, input_data, bundle):
    if not isinstance(input_data, Mapping):
        raise PredictionInputError(
            "input_data must be a dictionary of feature names and values."
        )

    required = list(bundle["required_columns"])
    supplied = set(input_data.keys())
    expected = set(required)

    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)

    errors = []

    if missing:
        errors.append(f"Missing fields: {missing}")

    if extra:
        errors.append(f"Unexpected fields: {extra}")

    if errors:
        raise PredictionInputError("; ".join(errors))

    preprocessor = bundle["pipeline"].named_steps["preprocessor"]
    numeric_columns, categorical_columns = _get_column_groups(
        preprocessor
    )

    for column in required:
        value = input_data[column]

        # Missing values are allowed when the fitted imputer supports them.
        if _is_missing(value):
            continue

        if column in numeric_columns:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise PredictionInputError(
                    f"'{column}' must be a number or null."
                )

            if not math.isfinite(float(value)):
                raise PredictionInputError(
                    f"'{column}' must be finite; NaN and infinity "
                    "are not valid numeric inputs."
                )

        elif column in categorical_columns:
            if isinstance(value, (dict, list, tuple, set)):
                raise PredictionInputError(
                    f"'{column}' must be a single categorical value or null."
                )

            if isinstance(value, str) and not value.strip():
                raise PredictionInputError(
                    f"'{column}' cannot be an empty string. "
                    "Use null if the value is unknown."
                )

    return pd.DataFrame(
        [[input_data[column] for column in required]],
        columns=required,
    )


def predict_risk(task, input_data):
    """
    Predict risk using one of the three packaged models.

    task:
        origination, early_warning_3m, or early_warning_6m

    input_data:
        Dictionary containing every required raw input field.
    """
    bundle = _load_bundle(task)
    row = _validate_input(task, input_data, bundle)

    score = float(bundle["pipeline"].predict_proba(row)[0, 1])
    threshold = float(bundle["threshold"])
    predicted_class = int(score >= threshold)

    return {
        "task": task,
        "purpose": PURPOSES[task],
        "predicted_score": round(score, 6),
        "validation_threshold": round(threshold, 4),
        "predicted_class": predicted_class,
        "interpretation": (
            "At or above the selected threshold"
            if predicted_class == 1
            else "Below the selected threshold"
        ),
        "score_note": (
            "Model score from a synthetic credit-risk dataset. "
            "It is not a validated real-world MSME default probability."
        ),
    }