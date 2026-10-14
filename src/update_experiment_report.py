
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main():
    source = REPORTS / "model_selection_cv_summary.csv"
    output = REPORTS / "experiment_tracking_report.csv"

    if not source.exists():
        raise FileNotFoundError(
            f"Missing {source}. Run "
            "python -m src.update_model_selection first."
        )

    cv = pd.read_csv(source)

    required = {
        "Model",
        "MAE_mean",
        "RMSE_mean",
        "WAPE_mean",
    }
    missing = required - set(cv.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns in CV report: {sorted(missing)}"
        )

    # Keep the fair, same-fold comparison clearly identified.
    cv["EvaluationType"] = "Same-Fold Rolling CV"
    cv["Recommendation"] = "Candidate"

    # Lower mean WAPE ranks higher; preserve variability columns.
    cv["WAPERank"] = cv["WAPE_mean"].rank(
        method="min", ascending=True
    )

    preferred_columns = [
        "Model",
        "EvaluationType",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "RMSE_std",
        "WAPE_mean",
        "WAPE_std",
        "MAE_median",
        "RMSE_median",
        "WAPE_median",
        "WAPERank",
        "Recommendation",
    ]
    report = cv[
        [column for column in preferred_columns if column in cv.columns]
    ].sort_values("WAPERank")

    report.to_csv(output, index=False)

    print("\nExperiment tracking report:")
    print(report.to_string(index=False))
    print(f"\nSaved to: {output}")
    print(
        "\nNote: this report contains same-fold historical validation "
        "results, not recursive multi-day forecast accuracy."
    )


if __name__ == "__main__":
    main()
