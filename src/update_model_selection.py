
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main():
    source_path = REPORTS / "model_comparison_cv_summary.csv"
    output_path = REPORTS / "model_selection_cv_summary.csv"

    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing {source_path}. Run "
            "python -m src.compare_models_cv first."
        )

    # This file was exported with a two-level column header
    # and Model stored as the index.
    summary = pd.read_csv(source_path, header=[0, 1], index_col=0)

    # Flatten the two-level headers, e.g. ("MAE", "mean") -> "MAE_mean".
    flattened_columns = []
    for top, bottom in summary.columns:
        top = str(top).strip()
        bottom = str(bottom).strip()

        if bottom.startswith("Unnamed") or bottom == "nan":
            flattened_columns.append(top)
        else:
            flattened_columns.append(f"{top}_{bottom}")

    summary.columns = flattened_columns
    summary.index.name = "Model"
    summary = summary.reset_index()

    # Convert metric columns to numeric values.
    for column in summary.columns:
        if column != "Model":
            summary[column] = pd.to_numeric(
                summary[column], errors="coerce"
            )

    summary.to_csv(output_path, index=False)

    print("\nClean model-selection CV summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
