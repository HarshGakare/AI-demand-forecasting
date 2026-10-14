
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

CV_PATH = REPORTS / "rolling_cv_results.csv"
EVALUATION_PATH = REPORTS / "champion_model_evaluation.csv"
SUMMARY_PATH = REPORTS / "champion_model_summary.csv"

METRICS = ["MAE", "RMSE", "WAPE"]


def normalize_name(name):
    """Normalize column names for case-insensitive matching."""
    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


def find_column(df, expected_name):
    """Find a column despite capitalization or underscore differences."""
    target = normalize_name(expected_name)

    for column in df.columns:
        if normalize_name(column) == target:
            return column

    return None


def load_cv_metrics():
    if not CV_PATH.exists():
        raise FileNotFoundError(
            f"Rolling CV results not found: {CV_PATH}"
        )

    df = pd.read_csv(CV_PATH)

    if df.empty:
        raise ValueError("Rolling CV results file is empty.")

    metric_columns = {}

    for metric in METRICS:
        column = find_column(df, metric)

        if column is None:
            raise ValueError(
                f"Could not find the {metric} column. "
                f"Available columns: {list(df.columns)}"
            )

        metric_columns[metric] = column

    metrics = {}

    for metric, column in metric_columns.items():
        values = pd.to_numeric(df[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()

        if values.empty:
            raise ValueError(
                f"No valid numeric values found for {metric}."
            )

        metrics[metric] = float(values.mean())

    print("Rolling cross-validation mean metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


def update_evaluation_report(metrics):
    if not EVALUATION_PATH.exists():
        raise FileNotFoundError(
            f"Champion evaluation report not found: {EVALUATION_PATH}"
        )

    df = pd.read_csv(EVALUATION_PATH)

    model_col = find_column(df, "Model")
    evaluation_col = find_column(df, "EvaluationType")

    if model_col is None or evaluation_col is None:
        raise ValueError(
            "Expected Model and EvaluationType columns. "
            f"Available columns: {list(df.columns)}"
        )

    row_mask = (
        df[model_col].astype(str).str.strip().eq("Tuned LightGBM")
        & df[evaluation_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("rolling cv mean")
    )

    if not row_mask.any():
        raise ValueError(
            "Could not find the Tuned LightGBM / Rolling CV Mean row."
        )

    for metric, value in metrics.items():
        column = find_column(df, metric)

        if column is None:
            raise ValueError(
                f"Missing {metric} column in champion evaluation report."
            )

        df.loc[row_mask, column] = value

    df.to_csv(EVALUATION_PATH, index=False)
    print(f"\nUpdated: {EVALUATION_PATH}")


def update_summary_report(metrics):
    if not SUMMARY_PATH.exists():
        print(
            "Summary report does not exist; skipping summary update."
        )
        return

    df = pd.read_csv(SUMMARY_PATH)

    if df.empty:
        print("Summary report is empty; skipping summary update.")
        return

    model_col = find_column(df, "Model")

    if model_col is None:
        print(
            "Could not identify a Model column in the summary. "
            "The evaluation report has been updated, "
            "but the summary was left unchanged."
        )
        return

    row_mask = (
        df[model_col]
        .astype(str)
        .str.strip()
        .eq("Tuned LightGBM")
    )

    if not row_mask.any():
        print(
            "Tuned LightGBM not found in the summary; "
            "the summary was left unchanged."
        )
        return

    updated = False

    for metric, value in metrics.items():
        # Update common summary naming conventions.
        candidates = [
            f"{metric}_mean",
            f"{metric}_Mean",
            f"Mean{metric}",
            f"{metric}Mean",
        ]

        column = None

        for candidate in candidates:
            column = find_column(df, candidate)
            if column is not None:
                break

        if column is not None:
            df.loc[row_mask, column] = value
            updated = True

    if updated:
        df.to_csv(SUMMARY_PATH, index=False)
        print(f"Updated: {SUMMARY_PATH}")
    else:
        print(
            "No recognizable mean-metric columns were found in the "
            "summary. It was left unchanged to avoid corrupting it."
        )


def main():
    metrics = load_cv_metrics()
    update_evaluation_report(metrics)
    update_summary_report(metrics)

    print("\nModel evaluation report repair completed.")
    print(
        "Note: this repairs metric aggregation only; "
        "it does not prove which model is best."
    )


if __name__ == "__main__":
    main()
