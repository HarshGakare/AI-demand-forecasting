
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_merged.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_merged.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_train.csv"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "data_quality_report.csv"


def validate_required_columns(df, required_columns, dataset_name):
    """Check that all required columns exist."""
    errors = []

    missing_columns = sorted(set(required_columns) - set(df.columns))

    if missing_columns:
        errors.append(
            f"{dataset_name}: missing required columns: {missing_columns}"
        )

    return errors


def validate_missing_values(df, dataset_name):
    """Report columns containing missing values."""
    issues = []

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    for column, count in missing_counts.items():
        issues.append(
            {
                "Dataset": dataset_name,
                "Check": "Missing values",
                "Column": column,
                "Count": int(count),
                "Severity": "WARNING",
                "Details": (
                    f"{count} missing values "
                    f"({count / len(df) * 100:.2f}%)"
                ),
            }
        )

    if missing_counts.empty:
        issues.append(
            {
                "Dataset": dataset_name,
                "Check": "Missing values",
                "Column": "ALL",
                "Count": 0,
                "Severity": "PASS",
                "Details": "No missing values found",
            }
        )

    return issues


def validate_duplicates(df, dataset_name, key_columns):
    """Check duplicate records using the specified business keys."""
    issues = []

    missing_keys = sorted(set(key_columns) - set(df.columns))

    if missing_keys:
        issues.append(
            {
                "Dataset": dataset_name,
                "Check": "Duplicate records",
                "Column": ",".join(key_columns),
                "Count": 0,
                "Severity": "ERROR",
                "Details": f"Cannot check duplicates; missing keys: {missing_keys}",
            }
        )
        return issues

    duplicate_count = int(df.duplicated(subset=key_columns).sum())

    issues.append(
        {
            "Dataset": dataset_name,
            "Check": "Duplicate records",
            "Column": ",".join(key_columns),
            "Count": duplicate_count,
            "Severity": "ERROR" if duplicate_count else "PASS",
            "Details": (
                f"{duplicate_count} duplicate key combinations"
                if duplicate_count
                else "No duplicate key combinations found"
            ),
        }
    )

    return issues


def validate_date_column(df, dataset_name):
    """Check that dates can be parsed."""
    issues = []

    if "Date" not in df.columns:
        issues.append(
            {
                "Dataset": dataset_name,
                "Check": "Date validity",
                "Column": "Date",
                "Count": 0,
                "Severity": "ERROR",
                "Details": "Required Date column is missing",
            }
        )
        return issues

    parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
    invalid_count = int(parsed_dates.isna().sum())

    issues.append(
        {
            "Dataset": dataset_name,
            "Check": "Date validity",
            "Column": "Date",
            "Count": invalid_count,
            "Severity": "ERROR" if invalid_count else "PASS",
            "Details": (
                f"{invalid_count} invalid or missing dates"
                if invalid_count
                else "All dates are valid"
            ),
        }
    )

    return issues


def validate_numeric_range(
    df,
    dataset_name,
    column,
    minimum=None,
    maximum=None,
    allow_missing=True,
):
    """Check whether numeric values fall within an expected range."""
    issues = []

    if column not in df.columns:
        return issues

    numeric_values = pd.to_numeric(df[column], errors="coerce")

    invalid_count = int(
        (df[column].notna() & numeric_values.isna()).sum()
    )

    if minimum is not None:
        invalid_count += int((numeric_values < minimum).sum())

    if maximum is not None:
        invalid_count += int((numeric_values > maximum).sum())

    missing_count = int(df[column].isna().sum())

    severity = "ERROR" if invalid_count else "PASS"

    if not allow_missing and missing_count:
        severity = "ERROR"

    details = (
        f"Invalid values: {invalid_count}; "
        f"missing values: {missing_count}; "
        f"expected range: {minimum} to {maximum}"
    )

    issues.append(
        {
            "Dataset": dataset_name,
            "Check": "Numeric range",
            "Column": column,
            "Count": invalid_count + (
                missing_count if not allow_missing else 0
            ),
            "Severity": severity,
            "Details": details,
        }
    )

    return issues


def validate_dataset(
    df,
    dataset_name,
    required_columns,
    key_columns=None,
    is_training_data=False,
):
    """Run data-quality checks on one dataset."""
    issues = []

    if df.empty:
        issues.append(
            {
                "Dataset": dataset_name,
                "Check": "Dataset not empty",
                "Column": "ALL",
                "Count": 0,
                "Severity": "ERROR",
                "Details": "Dataset contains zero rows",
            }
        )
        return issues

    issues.append(
        {
            "Dataset": dataset_name,
            "Check": "Dataset not empty",
            "Column": "ALL",
            "Count": len(df),
            "Severity": "PASS",
            "Details": f"Dataset contains {len(df):,} rows",
        }
    )

    issues.extend(
        validate_required_columns(
            df,
            required_columns,
            dataset_name,
        )
    )

    issues.extend(validate_missing_values(df, dataset_name))
    issues.extend(validate_date_column(df, dataset_name))

    if key_columns:
        issues.extend(
            validate_duplicates(
                df,
                dataset_name,
                key_columns,
            )
        )

    if "Store" in df.columns:
        issues.extend(
            validate_numeric_range(
                df,
                dataset_name,
                "Store",
                minimum=1,
                allow_missing=False,
            )
        )

    if "Sales" in df.columns:
        issues.extend(
            validate_numeric_range(
                df,
                dataset_name,
                "Sales",
                minimum=0,
                allow_missing=not is_training_data,
            )
        )

    if "Open" in df.columns:
        issues.extend(
            validate_numeric_range(
                df,
                dataset_name,
                "Open",
                minimum=0,
                maximum=1,
            )
        )

    if "Promo" in df.columns:
        issues.extend(
            validate_numeric_range(
                df,
                dataset_name,
                "Promo",
                minimum=0,
                maximum=1,
            )
        )

    return issues


def load_csv_if_exists(path, dataset_name):
    """Load a CSV safely and report file-loading failures."""
    if not path.exists():
        print(f"SKIPPED: {dataset_name} file not found: {path}")
        return None

    try:
        df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
        print(f"Loaded {dataset_name}: {len(df):,} rows")
        return df
    except Exception as exc:
        print(f"ERROR loading {dataset_name}: {exc}")
        return None


def run_data_quality_checks():
    """Validate available project datasets and save a report."""
    all_issues = []

    train_df = load_csv_if_exists(
        TRAIN_PATH,
        "Merged training data",
    )

    if train_df is not None:
        all_issues.extend(
            validate_dataset(
                train_df,
                dataset_name="train_merged",
                required_columns=[
                    "Store",
                    "Date",
                    "Sales",
                    "Customers",
                    "Open",
                    "Promo",
                ],
                key_columns=["Store", "Date"],
                is_training_data=True,
            )
        )

    test_df = load_csv_if_exists(
        TEST_PATH,
        "Merged test data",
    )

    if test_df is not None:
        all_issues.extend(
            validate_dataset(
                test_df,
                dataset_name="test_merged",
                required_columns=[
                    "Store",
                    "Date",
                    "Open",
                    "Promo",
                ],
                key_columns=["Store", "Date"],
                is_training_data=False,
            )
        )

    features_df = load_csv_if_exists(
        FEATURES_PATH,
        "Training features",
    )

    if features_df is not None:
        feature_required = [
            "Store",
            "Date",
            "Sales",
        ]

        all_issues.extend(
            validate_dataset(
                features_df,
                dataset_name="features_train",
                required_columns=feature_required,
                key_columns=["Store", "Date"],
                is_training_data=True,
            )
        )

    if not all_issues:
        print(
            "\nNo datasets were available to validate. "
            "Run the earlier data preparation steps first."
        )
        return False

    report_df = pd.DataFrame(all_issues)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH, index=False)

    error_count = int(
        (report_df["Severity"] == "ERROR").sum()
    )
    warning_count = int(
        (report_df["Severity"] == "WARNING").sum()
    )
    pass_count = int(
        (report_df["Severity"] == "PASS").sum()
    )

    print("\n" + "=" * 60)
    print("DATA QUALITY SUMMARY")
    print("=" * 60)
    print(f"Passed checks: {pass_count}")
    print(f"Warnings:      {warning_count}")
    print(f"Errors:        {error_count}")
    print(f"\nReport saved to: {REPORT_PATH}")

    problem_rows = report_df[
        report_df["Severity"].isin(["ERROR", "WARNING"])
    ]

    if not problem_rows.empty:
        print("\nIssues requiring attention:")
        print(
            problem_rows[
                [
                    "Dataset",
                    "Check",
                    "Column",
                    "Count",
                    "Severity",
                    "Details",
                ]
            ].to_string(index=False)
        )

    if error_count:
        print("\nRESULT: FAILED — fix errors before relying on the data.")
        return False

    print("\nRESULT: PASSED — no errors detected.")
    return True


if __name__ == "__main__":
    success = run_data_quality_checks()

    if not success:
        raise SystemExit(1)
