
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data" / "processed" / "test_merged.csv"
BACKUP_PATH = ROOT / "data" / "processed" / "test_merged_before_open_fix.csv"
AUDIT_PATH = ROOT / "reports" / "missing_open_imputation_audit.csv"


def main():
    test = pd.read_csv(
        TEST_PATH,
        parse_dates=["Date"],
        low_memory=False,
    )

    missing_mask = test["Open"].isna()
    missing_count = int(missing_mask.sum())

    if missing_count == 0:
        print("No missing Open values found. Nothing to change.")
        return

    # Preserve the original file before changing anything.
    if not BACKUP_PATH.exists():
        test.to_csv(BACKUP_PATH, index=False)
        print(f"Backup saved: {BACKUP_PATH}")

    # This inference is deliberately restricted to the known affected store.
    affected = missing_mask & (test["Store"] == 622)

    if int(affected.sum()) != missing_count:
        raise ValueError(
            "Missing Open values exist outside Store 622. "
            "Review them separately before proceeding."
        )

    audit = test.loc[
        affected,
        ["Id", "Store", "Date", "DayOfWeek", "Open"],
    ].copy()

    # Store 622 was historically open every Saturday and closed every Sunday.
    # On weekdays, use the historically dominant opening status.
    # Its weekday open rate exceeds 50% for every weekday.
    test.loc[affected, "Open"] = 1

    audit["Imputed_Open"] = 1
    audit["Imputation_Method"] = (
        "Store 622 historical weekday pattern; inferred, not verified"
    )

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    remaining_missing = int(test["Open"].isna().sum())

    print(f"Missing Open values originally: {missing_count}")
    print(f"Values inferred for Store 622: {int(affected.sum())}")
    print(f"Missing Open values remaining: {remaining_missing}")
    print(f"Audit report saved: {AUDIT_PATH}")
    print(f"Updated test data saved: {TEST_PATH}")

    if remaining_missing:
        print(
            "WARNING: Some Open values remain missing; "
            "do not generate final predictions yet."
        )
    else:
        print(
            "All test Open values are populated. "
            "The 11 inferred values remain documented in the audit report."
        )


if __name__ == "__main__":
    main()
