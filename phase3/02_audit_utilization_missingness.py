
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

FILES = {
    "3M": (
        "model_b_final_early_warning_3m.parquet",
        "early_warning_target_3m",
        "avg_utilization_3m",
        "max_utilization_3m",
    ),
    "6M": (
        "model_b_final_early_warning_6m.parquet",
        "early_warning_target_6m",
        "avg_utilization_6m",
        "max_utilization_6m",
    ),
}

for name, (filename, target, avg_col, max_col) in FILES.items():
    print("\n" + "=" * 75)
    print(f"UTILIZATION MISSINGNESS AUDIT — {name}")
    print("=" * 75)

    df = pd.read_parquet(DATA_DIR / filename)

    for col in [avg_col, max_col]:
        missing = df[col].isna()

        print(f"\nColumn: {col}")
        print(f"Missing: {missing.sum():,} / {len(df):,}")
        print(f"Missing percentage: {missing.mean() * 100:.2f}%")

        print("\nTarget distribution by missingness:")
        print(
            pd.crosstab(
                missing.map({
                    True: "Missing",
                    False: "Present"
                }),
                df[target],
                normalize="index"
            ).mul(100).round(2)
        )

    both_missing = df[[avg_col, max_col]].isna().all(axis=1)
    print(f"\nBoth utilization values missing: {both_missing.sum():,}")

    print("\nMissingness compared with payment observations:")
    print(
        df.groupby(both_missing)[
            [f"payment_observations_{name.lower()}",
             f"months_observed_{name.lower()}"]
        ].agg(["count", "mean", "median"]).round(2)
    )

print("\nAudit complete. No files were modified.")
