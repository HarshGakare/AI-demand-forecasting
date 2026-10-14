
import pandas as pd

from src.data_validation import (
    validate_dataset,
    validate_date_column,
    validate_duplicates,
    validate_numeric_range,
    validate_required_columns,
)


def test_required_columns_detects_missing_column():
    df = pd.DataFrame({"Store": [1], "Date": ["2015-01-01"]})

    issues = validate_required_columns(
        df,
        ["Store", "Date", "Sales"],
        "test",
    )

    assert len(issues) == 1
    assert "Sales" in issues[0]


def test_valid_dates_pass():
    df = pd.DataFrame({"Date": ["2015-01-01", "2015-01-02"]})

    issues = validate_date_column(df, "test")

    assert issues[0]["Severity"] == "PASS"
    assert issues[0]["Count"] == 0


def test_invalid_dates_are_detected():
    df = pd.DataFrame({"Date": ["2015-01-01", "not-a-date"]})

    issues = validate_date_column(df, "test")

    assert issues[0]["Severity"] == "ERROR"
    assert issues[0]["Count"] == 1


def test_duplicate_store_date_records_are_detected():
    df = pd.DataFrame(
        {
            "Store": [1, 1, 2],
            "Date": ["2015-01-01", "2015-01-01", "2015-01-01"],
        }
    )

    issues = validate_duplicates(
        df,
        "test",
        ["Store", "Date"],
    )

    assert issues[0]["Severity"] == "ERROR"
    assert issues[0]["Count"] == 1


def test_non_duplicate_store_date_records_pass():
    df = pd.DataFrame(
        {
            "Store": [1, 1, 2],
            "Date": ["2015-01-01", "2015-01-02", "2015-01-01"],
        }
    )

    issues = validate_duplicates(
        df,
        "test",
        ["Store", "Date"],
    )

    assert issues[0]["Severity"] == "PASS"
    assert issues[0]["Count"] == 0


def test_negative_sales_are_detected():
    df = pd.DataFrame({"Sales": [100, -5, 50]})

    issues = validate_numeric_range(
        df,
        "test",
        "Sales",
        minimum=0,
    )

    assert issues[0]["Severity"] == "ERROR"
    assert issues[0]["Count"] == 1


def test_valid_sales_pass():
    df = pd.DataFrame({"Sales": [0, 100, 500]})

    issues = validate_numeric_range(
        df,
        "test",
        "Sales",
        minimum=0,
    )

    assert issues[0]["Severity"] == "PASS"
    assert issues[0]["Count"] == 0


def test_numeric_check_detects_non_numeric_values():
    df = pd.DataFrame({"Store": [1, "invalid", 3]})

    issues = validate_numeric_range(
        df,
        "test",
        "Store",
        minimum=1,
        allow_missing=False,
    )

    assert issues[0]["Severity"] == "ERROR"
    assert issues[0]["Count"] == 1


def test_missing_required_date_is_detected():
    df = pd.DataFrame({"Store": [1]})

    issues = validate_date_column(df, "test")

    assert issues[0]["Severity"] == "ERROR"


def test_missing_sales_are_allowed_when_configured():
    df = pd.DataFrame({"Sales": [100, None, 50]})

    issues = validate_numeric_range(
        df,
        "test",
        "Sales",
        minimum=0,
        allow_missing=True,
    )

    assert issues[0]["Severity"] == "PASS"
    assert issues[0]["Count"] == 0


def test_missing_sales_can_be_rejected():
    df = pd.DataFrame({"Sales": [100, None, 50]})

    issues = validate_numeric_range(
        df,
        "test",
        "Sales",
        minimum=0,
        allow_missing=False,
    )

    assert issues[0]["Severity"] == "ERROR"
    assert issues[0]["Count"] == 1


def test_dataset_validation_detects_duplicate_records():
    df = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": ["2015-01-01", "2015-01-01"],
            "Sales": [100, 200],
            "Customers": [10, 20],
            "Open": [1, 1],
            "Promo": [0, 0],
        }
    )

    issues = validate_dataset(
        df,
        dataset_name="test",
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

    assert any(
        issue["Check"] == "Duplicate records"
        and issue["Severity"] == "ERROR"
        for issue in issues
    )
