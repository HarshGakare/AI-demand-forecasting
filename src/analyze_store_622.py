
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "train_merged.csv"


def main():
    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["Date"],
        low_memory=False,
    )

    store = train.loc[train["Store"] == 622].copy()

    # Compare historical opening status by day of week.
    summary = (
        store.groupby("DayOfWeek")
        .agg(
            Records=("Open", "size"),
            Open_Days=("Open", lambda x: (x == 1).sum()),
            Closed_Days=("Open", lambda x: (x == 0).sum()),
            Missing_Open=("Open", lambda x: x.isna().sum()),
            Mean_Sales_When_Open=(
                "Sales",
                lambda x: x[store.loc[x.index, "Open"] == 1].mean(),
            ),
        )
        .reset_index()
    )

    summary["Open_Rate"] = (
        summary["Open_Days"]
        / (summary["Open_Days"] + summary["Closed_Days"])
        * 100
    )

    print("Store 622 historical opening pattern by weekday:")
    print(summary.round(2).to_string(index=False))

    print("\nInterpretation:")
    for _, row in summary.iterrows():
        print(
            f"DayOfWeek {int(row['DayOfWeek'])}: "
            f"{int(row['Open_Days'])} open, "
            f"{int(row['Closed_Days'])} closed, "
            f"open rate {row['Open_Rate']:.2f}%"
        )


if __name__ == "__main__":
    main()
