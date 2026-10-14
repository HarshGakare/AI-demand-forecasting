
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "features_train.csv"
TEST_PATH = PROCESSED_DIR / "features_test.csv"


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print("Training features:", train.shape)
    print("Test features:", test.shape)

    missing_open = int(test["Open"].isna().sum())
    print("Missing Open values in saved test features:", missing_open)

    assert train.shape == (978184, 45), "Unexpected training dimensions"
    assert test.shape == (41088, 47), "Unexpected test dimensions"
    assert missing_open == 0, "Missing Open values remain in test features"

    assert set(test["Open"].dropna().unique()).issubset({0, 1}), (
        "Open contains unexpected values"
    )

    print("\nPASS: Saved feature outputs verified.")


if __name__ == "__main__":
    main()
