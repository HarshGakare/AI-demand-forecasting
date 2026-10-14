
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

REQUIRED_REPORTS = [
    "model_comparison_cv.csv",
    "model_comparison_cv_summary.csv",
    "model_selection_cv_summary.csv",
    "experiment_tracking_report.csv",
    "champion_model_evaluation.csv",
    "rolling_cv_results.csv",
    "rolling_cv_summary.csv",
    "multi_horizon_metrics.csv",
    "recursive_forecast_metrics.csv",
    "inventory_summary.csv",
    "data_quality_report.csv",
    "shap_feature_importance.csv",
]


def main():
    failures = []

    print("Project report integrity check\n")

    for filename in REQUIRED_REPORTS:
        path = REPORTS / filename

        if not path.exists():
            print(f"MISSING  {filename}")
            failures.append(filename)
            continue

        try:
            df = pd.read_csv(path)

            if df.empty:
                print(f"EMPTY    {filename}")
                failures.append(filename)
            else:
                print(
                    f"OK       {filename}: "
                    f"{len(df):,} rows, {len(df.columns)} columns"
                )
        except Exception as exc:
            print(f"INVALID  {filename}: {exc}")
            failures.append(filename)

    print(f"\nReports checked: {len(REQUIRED_REPORTS)}")
    print(f"Issues found: {len(failures)}")

    if failures:
        print("\nReview these reports:")
        for filename in failures:
            print(f" - {filename}")
    else:
        print("All required reports exist and contain readable data.")


if __name__ == "__main__":
    main()
