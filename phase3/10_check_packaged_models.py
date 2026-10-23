
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = ROOT / "models" / "phase3" / "packaged"

BUNDLES = [
    "model_a_bundle.joblib",
    "model_b_3m_bundle.joblib",
    "model_b_6m_bundle.joblib",
]

for filename in BUNDLES:
    path = BUNDLE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing bundle: {path}")

    bundle = joblib.load(path)
    pipeline = bundle["pipeline"]

    assert bundle["required_columns"], "No input schema found."
    assert 0 < bundle["threshold"] < 1
    assert hasattr(pipeline, "predict_proba")

    print(f"\n{filename}")
    print(f"  Task: {bundle['task']}")
    print(f"  Required input fields: {len(bundle['required_columns'])}")
    print(f"  Validation threshold: {bundle['threshold']}")
    print("  Probability prediction interface: available")

print("\nAll packaged model bundles passed basic artifact checks.")
