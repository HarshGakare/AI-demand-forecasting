
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_merged.csv"


def main():
    df = pd.read_csv(TEST_PATH, low_memory=False)

    print("File:", TEST_PATH)
    print("Rows:", len(df))
    print("Missing Open values:", df["Open"].isna().sum())

    missing = df[df["Open"].isna()]

    if not missing.empty:
        print("\nRows with missing Open:")
        print(missing[["Store", "Date", "Open"]].to_string(index=False))
    else:
        print("\nPASS: No missing Open values in test_merged.csv")


if __name__ == "__main__":
    main()
