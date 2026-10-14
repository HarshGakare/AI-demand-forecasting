
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data" / "processed" / "test_merged.csv"
BACKUP_PATH = ROOT / "data" / "processed" / "test_merged_before_open_fix.csv"
AUDIT_PATH = ROOT / "reports" / "missing_open_imputation_audit.csv"


def main():
    test = pd.read_csv(TEST_PATH, low_memory=False)
    backup = pd.read_csv(BACKUP_PATH, low_memory=False)
    audit = pd.read_csv(AUDIT_PATH)

    print("=== Open Status Fix Verification ===")
    print(f"Updated test rows: {len(test):,}")
    print(f"Backup test rows: {len(backup):,}")
    print(f"Missing Open values now: {test['Open'].isna().sum()}")
    print(f"Invalid Open values: {(~test['Open'].isin([0, 1])).sum()}")
    print(f"Audit rows: {len(audit)}")

    checks = {
        "Test row count unchanged": len(test) == len(backup),
        "No missing Open values": test["Open"].notna().all(),
        "Open values are only 0 or 1": test["Open"].isin([0, 1]).all(),
        "Exactly 11 audited imputations": len(audit) == 11,
        "All audited rows belong to Store 622": (audit["Store"] == 622).all(),
        "Backup exists": BACKUP_PATH.exists(),
    }

    print("\n=== Verification Checks ===")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    if not all(checks.values()):
        raise RuntimeError("Verification failed. Review the output above.")

    print("\nSUCCESS: Test data passed all opening-status checks.")


if __name__ == "__main__":
    main()
