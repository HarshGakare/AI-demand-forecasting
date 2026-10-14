
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "Raw test data": ROOT / "data" / "raw" / "test.csv",
    "Current merged test data": ROOT / "data" / "processed" / "test_merged.csv",
    "Pre-fix merged backup": ROOT / "data" / "processed" / "test_merged_before_open_fix.csv",
    "Imputation audit": ROOT / "reports" / "missing_open_imputation_audit.csv",
}


def main():
    for label, path in FILES.items():
        print(f"\n--- {label} ---")

        if not path.exists():
            print(f"NOT FOUND: {path}")
            continue

        df = pd.read_csv(path, low_memory=False)

        if "Open" not in df.columns:
            print("No Open column in this file.")
            continue

        print(f"Rows: {len(df):,}")
        print(f"Missing Open: {df['Open'].isna().sum()}")

        if "Store" in df.columns:
            missing = df.loc[df["Open"].isna()]
            if not missing.empty:
                print("Stores with missing Open:")
                print(missing["Store"].value_counts().to_string())


if __name__ == "__main__":
    main()
