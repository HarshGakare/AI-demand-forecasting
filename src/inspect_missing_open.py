
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "processed" / "test_merged.csv"
STORE_PATH = ROOT / "data" / "raw" / "store.csv"
OUTPUT_PATH = ROOT / "reports" / "missing_open_rows.csv"


def main():
    test = pd.read_csv(TEST_PATH, parse_dates=["Date"])
    stores = pd.read_csv(STORE_PATH)

    missing = test.loc[test["Open"].isna()].copy()

    if missing.empty:
        print("No missing Open values found.")
        return

    # Attach store metadata to help inspect the affected records.
    store_columns = [
        column for column in [
            "Store",
            "StoreType",
            "Assortment",
            "Promo2",
        ]
        if column in stores.columns
    ]

    missing = missing.merge(
        stores[store_columns],
        on="Store",
        how="left",
        suffixes=("", "_store"),
        validate="many_to_one",
    )

    display_columns = [
        column for column in [
            "Id",
            "Store",
            "Date",
            "DayOfWeek",
            "Promo",
            "StateHoliday",
            "SchoolHoliday",
            "Open",
            "StoreType",
            "Assortment",
        ]
        if column in missing.columns
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(OUTPUT_PATH, index=False)

    print(f"Missing Open rows: {len(missing)}")
    print(missing[display_columns].to_string(index=False))
    print(f"\nSaved inspection file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
