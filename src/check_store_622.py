
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "train_merged.csv"
TEST_PATH = ROOT / "data" / "processed" / "test_merged.csv"


def main():
    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["Date"],
        low_memory=False,
    )
    test = pd.read_csv(
        TEST_PATH,
        parse_dates=["Date"],
        low_memory=False,
    )

    store_history = train.loc[
        train["Store"] == 622,
        ["Store", "Date", "DayOfWeek", "Promo", "StateHoliday", "Open", "Sales"],
    ].sort_values("Date")

    print("Store 622 historical opening-status counts:")
    print(store_history["Open"].value_counts(dropna=False).sort_index())

    print("\nMost recent historical records:")
    print(store_history.tail(20).to_string(index=False))

    missing_dates = test.loc[
        (test["Store"] == 622) & test["Open"].isna(),
        ["Id", "Store", "Date", "DayOfWeek", "Promo", "StateHoliday"],
    ].sort_values("Date")

    print("\nTest dates with missing Open:")
    print(missing_dates.to_string(index=False))


if __name__ == "__main__":
    main()
