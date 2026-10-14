
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

OUTPUT_PATH = REPORTS / "experiment_tracking_report.csv"

INPUT_FILES = {
    "rolling_cv": REPORTS / "rolling_cv_results.csv",
    "holdout": REPORTS / "lightgbm_metrics.csv",
    "best_params": REPORTS / "lightgbm_best_params.csv",
    "baseline": REPORTS / "baseline_metrics.csv",
    "shap": REPORTS / "shap_feature_importance.csv",
}


def read_csv_if_available(path):
    if not path.exists():
        print(f"WARNING: Report not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
        print(f"Loaded {path.name}: {len(df)} rows")
        return df
    except Exception as exc:
        print(f"WARNING: Could not read {path.name}: {exc}")
        return None


def normalize_column_name(name):
    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


def find_column(df, candidates):
    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        match = normalized.get(normalize_column_name(candidate))
        if match is not None:
            return match

    return None


def add_record(records, section, name, value, source):
    records.append(
        {
            "Section": section,
            "Name": str(name),
            "Value": value,
            "Source": source,
        }
    )


def collect_cv_records(records):
    path = INPUT_FILES["rolling_cv"]
    df = read_csv_if_available(path)

    if df is None or df.empty:
        return

    for metric in ["MAE", "RMSE", "WAPE"]:
        column = find_column(df, [metric])

        if column is None:
            continue

        values = pd.to_numeric(df[column], errors="coerce").dropna()

        if values.empty:
            continue

        add_record(
            records,
            "Rolling CV",
            f"{metric}_mean",
            round(float(values.mean()), 6),
            path.name,
        )

        add_record(
            records,
            "Rolling CV",
            f"{metric}_std",
            round(float(values.std(ddof=0)), 6),
            path.name,
        )


def collect_holdout_records(records):
    path = INPUT_FILES["holdout"]
    df = read_csv_if_available(path)

    if df is None or df.empty:
        return

    # Supports either a row containing MAE/RMSE/WAPE or a
    # two-column Metric/Value report.
    metric_column = find_column(df, ["Metric"])
    value_column = find_column(df, ["Value"])

    if metric_column and value_column:
        for _, row in df.iterrows():
            metric = str(row[metric_column])
            value = pd.to_numeric(
                pd.Series([row[value_column]]),
                errors="coerce",
            ).iloc[0]

            if pd.notna(value):
                add_record(
                    records,
                    "Final Holdout",
                    metric,
                    round(float(value), 6),
                    path.name,
                )
        return

    for metric in ["MAE", "RMSE", "WAPE"]:
        column = find_column(df, [metric])

        if column is None:
            continue

        values = pd.to_numeric(df[column], errors="coerce").dropna()

        for index, value in enumerate(values):
            add_record(
                records,
                "Final Holdout",
                f"{metric}_row_{index + 1}",
                round(float(value), 6),
                path.name,
            )


def collect_best_params(records):
    path = INPUT_FILES["best_params"]
    df = read_csv_if_available(path)

    if df is None or df.empty:
        return

    if len(df.columns) >= 2:
        key_col = df.columns[0]
        value_col = df.columns[1]

        for _, row in df.iterrows():
            if pd.isna(row[key_col]):
                continue

            add_record(
                records,
                "Best Hyperparameters",
                row[key_col],
                row[value_col],
                path.name,
            )


def collect_baseline_records(records):
    path = INPUT_FILES["baseline"]
    df = read_csv_if_available(path)

    if df is None or df.empty:
        return

    model_col = find_column(df, ["Model", "ModelName"])

    if model_col is None:
        model_col = df.columns[0]

    for _, row in df.iterrows():
        model_name = str(row[model_col])

        for metric in ["MAE", "RMSE", "WAPE"]:
            column = find_column(df, [metric])

            if column is None:
                continue

            value = pd.to_numeric(
                pd.Series([row[column]]),
                errors="coerce",
            ).iloc[0]

            if pd.notna(value):
                add_record(
                    records,
                    "Baseline Comparison",
                    f"{model_name}_{metric}",
                    round(float(value), 6),
                    path.name,
                )


def collect_shap_records(records):
    path = INPUT_FILES["shap"]
    df = read_csv_if_available(path)

    if df is None or df.empty:
        return

    feature_col = find_column(df, ["Feature"])
    importance_col = find_column(
        df,
        ["MeanAbsoluteSHAP", "MeanAbsSHAP"],
    )

    if feature_col is None or importance_col is None:
        print(
            "WARNING: Could not identify feature and SHAP importance columns."
        )
        return

    df[importance_col] = pd.to_numeric(
        df[importance_col],
        errors="coerce",
    )

    df = df.dropna(subset=[importance_col])
    df = df.sort_values(importance_col, ascending=False).head(15)

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        add_record(
            records,
            "SHAP Feature Importance",
            f"{rank}_{row[feature_col]}",
            round(float(row[importance_col]), 6),
            path.name,
        )


def main():
    records = []

    collect_cv_records(records)
    collect_holdout_records(records)
    collect_best_params(records)
    collect_baseline_records(records)
    collect_shap_records(records)

    if not records:
        raise RuntimeError(
            "No experiment data could be collected. "
            "Check that the source report files exist."
        )

    report = pd.DataFrame(records)
    REPORTS.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print("\nExperiment tracking report created.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Records: {len(report)}")
    print("\nRecord counts by section:")
    print(report.groupby("Section").size().to_string())


if __name__ == "__main__":
    main()
