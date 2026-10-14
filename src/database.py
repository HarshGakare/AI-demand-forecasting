import os
from pathlib import Path

import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from mysql.connector import Error
import numpy as np



PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

DB_HOST = os.getenv("MYSQL_HOST", "localhost")
DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "MYSQL_PASSWORD")
DB_NAME = os.getenv("MYSQL_DATABASE", "demand_forecasting")



if not DB_NAME.replace("_", "").isalnum():
    raise ValueError(
        "MYSQL_DATABASE must contain only letters, numbers, and underscores."
    )


def get_server_connection():
    """Connect to MySQL without selecting a database."""

    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        connection_timeout=10,
    )


def get_connection():
    """Connect to the project database."""

    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        connection_timeout=10,
    )



TABLES = {
    "forecast_predictions": """
        CREATE TABLE IF NOT EXISTS forecast_predictions (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            store_id INT NOT NULL,
            forecast_date DATE NOT NULL,
            actual_sales DOUBLE NULL,
            predicted_sales DOUBLE NOT NULL,
            horizon_days INT NULL,
            forecast_origin DATE NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_forecast_store_date (store_id, forecast_date),
            INDEX idx_forecast_horizon (horizon_days)
        )
    """,

    "inventory_recommendations": """
        CREATE TABLE IF NOT EXISTS inventory_recommendations (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            store_id INT NOT NULL,
            average_daily_demand DOUBLE NULL,
            forecast_daily_demand DOUBLE NULL,
            safety_stock DOUBLE NULL,
            reorder_point DOUBLE NULL,
            target_inventory DOUBLE NULL,
            current_inventory_proxy DOUBLE NULL,
            recommended_order_quantity DOUBLE NULL,
            inventory_status VARCHAR(50) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_inventory_store (store_id)
        )
    """,

    "model_metrics": """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(150) NOT NULL,
            evaluation_type VARCHAR(100) NOT NULL,
            mae DOUBLE NULL,
            rmse DOUBLE NULL,
            wape DOUBLE NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_metrics_model (model_name)
        )
    """,
}


def initialize_database():
    """Create the database and its project tables."""

    connection = None
    cursor = None

    try:
        connection = get_server_connection()
        cursor = connection.cursor()

        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
            "CHARACTER SET utf8mb4 "
            "COLLATE utf8mb4_unicode_ci"
        )

        connection.commit()
        print(f"Database ready: {DB_NAME}")

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        for table_name, create_statement in TABLES.items():
            cursor.execute(create_statement)
            print(f"Table ready: {table_name}")

        connection.commit()
        print("Database initialization completed.")

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()




def _value_or_none(value):
    """Convert pandas missing values to Python None for SQL NULL."""

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def _insert_dataframe(
    table_name,
    columns,
    rows,
    placeholders,
):
    """Insert rows using parameterized SQL statements."""

    if table_name not in TABLES:
        raise ValueError(f"Unsupported table: {table_name}")

    if not rows:
        print(f"No rows to insert into {table_name}.")
        return 0

    column_sql = ", ".join(f"`{column}`" for column in columns)

    sql = (
        f"INSERT INTO `{table_name}` ({column_sql}) "
        f"VALUES ({placeholders})"
    )

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.executemany(sql, rows)
        connection.commit()

        inserted = cursor.rowcount
        print(f"Inserted {inserted:,} rows into {table_name}.")

        return inserted

    except Error:
        if connection is not None and connection.is_connected():
            connection.rollback()
        raise

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()




def save_forecasts(csv_path=None):
    """Save multi-horizon forecast records to MySQL."""

    path = (
        Path(csv_path)
        if csv_path
        else PROJECT_ROOT
        / "reports"
        / "multi_horizon_predictions.csv"
    )

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Forecast report not found: {path}")

    df = pd.read_csv(path)

    required = {"Store", "Date"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Forecast CSV missing columns: {sorted(missing)}")

    # Support the column names used by the project's reports.
    prediction_col = next(
        (
            column
            for column in ["Prediction", "PredictedSales", "Forecast"]
            if column in df.columns
        ),
        None,
    )

    actual_col = next(
        (
            column
            for column in ["Sales", "ActualSales", "Actual"]
            if column in df.columns
        ),
        None,
    )

    if prediction_col is None:
        raise ValueError(
            "Forecast CSV must contain Prediction, PredictedSales, "
            "or Forecast."
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "ForecastOrigin" in df.columns:
        df["ForecastOrigin"] = pd.to_datetime(
            df["ForecastOrigin"],
            errors="coerce",
        )

    df = df.dropna(subset=["Store", "Date", prediction_col]).copy()

    columns = [
        "store_id",
        "forecast_date",
        "actual_sales",
        "predicted_sales",
        "horizon_days",
        "forecast_origin",
    ]

    rows = []

    for _, row in df.iterrows():
        rows.append(
            (
                int(row["Store"]),
                row["Date"].date(),
                (
                    _value_or_none(row[actual_col])
                    if actual_col
                    else None
                ),
                float(row[prediction_col]),
                (
                    int(row["Horizon"])
                    if "Horizon" in df.columns
                    and pd.notna(row.get("Horizon"))
                    else None
                ),
                (
                    row["ForecastOrigin"].date()
                    if "ForecastOrigin" in df.columns
                    and pd.notna(row.get("ForecastOrigin"))
                    else None
                ),
            )
        )

    return _insert_dataframe(
        "forecast_predictions",
        columns,
        rows,
        placeholders=",".join(["%s"] * len(columns)),
    )




def save_inventory_recommendations(csv_path=None):
    """Save inventory recommendation records to MySQL."""

    path = (
        Path(csv_path)
        if csv_path
        else PROJECT_ROOT
        / "reports"
        / "inventory_optimization.csv"
    )

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Inventory report not found: {path}")

    df = pd.read_csv(path)

    if "Store" not in df.columns:
        raise ValueError("Inventory CSV must contain a Store column.")

    # Normalize optional column names.
    aliases = {
        "AverageDailyDemand": "average_daily_demand",
        "ForecastDailyDemand": "forecast_daily_demand",
        "SafetyStock": "safety_stock",
        "ReorderPoint": "reorder_point",
        "TargetInventory": "target_inventory",
        "CurrentInventoryProxy": "current_inventory_proxy",
        "RecommendedOrderQuantity": "recommended_order_quantity",
        "InventoryStatus": "inventory_status",
        "ReorderStatus": "inventory_status",
        "Status": "inventory_status",
    }

    rename_map = {}

    for source, destination in aliases.items():
        if source in df.columns and destination not in df.columns:
            rename_map[source] = destination

    df = df.rename(columns=rename_map)

    numeric_columns = [
        "average_daily_demand",
        "forecast_daily_demand",
        "safety_stock",
        "reorder_point",
        "target_inventory",
        "current_inventory_proxy",
        "recommended_order_quantity",
    ]

    for column in numeric_columns:
        if column not in df.columns:
            df[column] = None

    if "inventory_status" not in df.columns:
        df["inventory_status"] = None

    df["Store"] = pd.to_numeric(df["Store"], errors="coerce")
    df = df.dropna(subset=["Store"]).copy()

    columns = [
        "store_id",
        *numeric_columns,
        "inventory_status",
    ]

    rows = []

    for _, row in df.iterrows():
        values = [int(row["Store"])]

        for column in numeric_columns:
            value = pd.to_numeric(
                pd.Series([row[column]]),
                errors="coerce",
            ).iloc[0]

            values.append(_value_or_none(value))

        status = row["inventory_status"]

        if pd.isna(status):
            status = None
        else:
            status = str(status)[:50]

        values.append(status)
        rows.append(tuple(values))

    return _insert_dataframe(
        "inventory_recommendations",
        columns,
        rows,
        placeholders=",".join(["%s"] * len(columns)),
    )



def save_model_metrics(csv_path=None):
    """Save model evaluation metrics to MySQL."""

    path = (
        Path(csv_path)
        if csv_path
        else PROJECT_ROOT
        / "reports"
        / "lightgbm_metrics.csv"
    )

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Model metrics report not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        print("Model metrics file is empty; nothing to save.")
        return 0

    # Support either a row-oriented report or one row of metric columns.
    lower_to_original = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    if {"mae", "rmse", "wape"}.issubset(lower_to_original):
        model_col = lower_to_original.get("model")
        evaluation_col = lower_to_original.get("evaluationtype")

        rows = []

        for _, row in df.iterrows():
            rows.append(
                (
                    str(row[model_col]) if model_col else "Tuned LightGBM",
                    str(row[evaluation_col]) if evaluation_col else "Final Holdout",
                    _value_or_none(
                        pd.to_numeric(
                            pd.Series([row[lower_to_original["mae"]]]),
                            errors="coerce",
                        ).iloc[0]
                    ),
                    _value_or_none(
                        pd.to_numeric(
                            pd.Series([row[lower_to_original["rmse"]]]),
                            errors="coerce",
                        ).iloc[0]
                    ),
                    _value_or_none(
                        pd.to_numeric(
                            pd.Series([row[lower_to_original["wape"]]]),
                            errors="coerce",
                        ).iloc[0]
                    ),
                )
            )

    else:
        # The training script may save metric names as rows.
        metric_col = lower_to_original.get("metric")
        value_col = lower_to_original.get("value")

        if metric_col is None or value_col is None:
            raise ValueError(
                "Unsupported metrics CSV structure. Expected MAE, RMSE, "
                "and WAPE columns, or Metric and Value columns."
            )

        metric_map = {}

        for _, row in df.iterrows():
            metric_map[str(row[metric_col]).strip().upper()] = (
                pd.to_numeric(
                    pd.Series([row[value_col]]),
                    errors="coerce",
                ).iloc[0]
            )

        rows = [
            (
                "Tuned LightGBM",
                "Final Holdout",
                _value_or_none(metric_map.get("MAE", np.nan)),
                _value_or_none(metric_map.get("RMSE", np.nan)),
                _value_or_none(metric_map.get("WAPE", np.nan)),
            )
        ]

    return _insert_dataframe(
        "model_metrics",
        [
            "model_name",
            "evaluation_type",
            "mae",
            "rmse",
            "wape",
        ],
        rows,
        placeholders="%s,%s,%s,%s,%s",
    )




def verify_database():
    """Display table names and row counts."""

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        print(f"\nConnected to database: {DB_NAME}")

        for table_name in TABLES:
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            count = cursor.fetchone()[0]
            print(f"{table_name}: {count:,} rows")

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()



def main():
    """Initialize and verify the project database."""

    if not DB_PASSWORD:
        raise ValueError(
            "MYSQL_PASSWORD is empty. Set it in your project-root .env file."
        )

    try:
        initialize_database()
        verify_database()

        print("\nMySQL setup completed successfully.")
        print(
            "The tables are ready. CSV import is a separate operation; "
            "run the save functions when you want to import reports."
        )

    except Error as exc:
        print(f"\nMySQL error: {exc}")
        print(
            "Check that MySQL Server is running and that your .env "
            "credentials are correct. The configured user must have "
            "permission to create the database and tables."
        )
        raise


if __name__ == "__main__":
    main()
    