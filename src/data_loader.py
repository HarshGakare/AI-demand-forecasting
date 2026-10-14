
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_data():
    """Load Rossmann CSV files from data/raw."""

    required_files = [
        "train.csv",
        "test.csv",
        "store.csv",
        "sample_submission.csv",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (RAW_DIR / filename).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing files in {RAW_DIR}: {missing_files}"
        )

    train = pd.read_csv(
    RAW_DIR / "train.csv",
    parse_dates=["Date"],
    low_memory=False,
    )

    test = pd.read_csv(
        RAW_DIR / "test.csv",
        parse_dates=["Date"],
        low_memory=False,
    )
    store = pd.read_csv(
        RAW_DIR / "store.csv",
    )

    sample_submission = pd.read_csv(
        RAW_DIR / "sample_submission.csv",
    )

    return train, test, store, sample_submission


def validate_columns(train, test, store):
    """Check that the expected columns exist."""

    expected_train = {
        "Store",
        "DayOfWeek",
        "Date",
        "Sales",
        "Customers",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
    }

    expected_test = {
        "Store",
        "DayOfWeek",
        "Date",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
    }

    expected_store = {
        "Store",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
    }

    checks = [
        ("train", train, expected_train),
        ("test", test, expected_test),
        ("store", store, expected_store),
    ]

    for name, dataframe, expected in checks:
        missing = expected - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"{name} is missing columns: {sorted(missing)}"
            )


def merge_store_data(train, test, store):
    """Merge store characteristics into train and test."""

    train_merged = train.merge(
        store,
        on="Store",
        how="left",
        validate="many_to_one",
    )

    test_merged = test.merge(
        store,
        on="Store",
        how="left",
        validate="many_to_one",
    )

    return train_merged, test_merged


def inspect_data(name, dataframe):
    """Print a compact data-quality report."""

    print(f"\n{'=' * 60}")
    print(f"DATASET: {name}")
    print(f"{'=' * 60}")

    print("\nShape:")
    print(dataframe.shape)

    print("\nData types:")
    print(dataframe.dtypes)

    print("\nMissing values:")
    print(
        dataframe.isna()
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )

    print("\nDuplicate rows:")
    print(dataframe.duplicated().sum())

    print("\nPreview:")
    print(dataframe.head())


def save_processed_data(train_merged, test_merged):
    """Save merged datasets after applying documented data fixes."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = PROCESSED_DIR / "train_merged.csv"
    test_path = PROCESSED_DIR / "test_merged.csv"

    # Handle known missing Open values for Store 622.
    # This is an inference based on historical weekday patterns,
    # not verified ground truth.
    missing_open = test_merged["Open"].isna()

    if missing_open.any():
        affected = missing_open & (test_merged["Store"] == 622)
        unexpected = missing_open & (test_merged["Store"] != 622)

        if unexpected.any():
            stores = sorted(
                test_merged.loc[unexpected, "Store"].unique().tolist()
            )
            raise ValueError(
                f"Unexpected missing Open values for stores: {stores}. "
                "Investigate before imputing."
            )

        test_merged.loc[affected, "Open"] = 1

        print(
            "Inferred Open=1 for "
            f"{int(affected.sum())} missing test rows belonging to Store 622."
        )

    # Save only after the imputation has been applied.
    train_merged.to_csv(
        train_path,
        index=False,
    )

    test_merged.to_csv(
        test_path,
        index=False,
    )

    print(f"\nSaved: {train_path}")
    print(f"Saved: {test_path}")
    print(
        "Missing Open values after imputation:",
        int(test_merged["Open"].isna().sum()),
    )

def main():
    train, test, store, sample_submission = load_raw_data()

    validate_columns(
        train,
        test,
        store,
    )

    train_merged, test_merged = merge_store_data(
        train,
        test,
        store,
    )

    inspect_data("TRAIN", train_merged)
    inspect_data("TEST", test_merged)
    inspect_data("STORE", store)
    inspect_data("SAMPLE SUBMISSION", sample_submission)

    save_processed_data(
        train_merged,
        test_merged,
    )


if __name__ == "__main__":
    main()
