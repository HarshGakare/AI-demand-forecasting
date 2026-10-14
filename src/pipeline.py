
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_STEPS = [
    {
        "name": "Data loading and merging",
        "module": "src.data_loader",
    },
    {
        "name": "Feature engineering",
        "module": "src.feature_engineering",
    },
    {
        "name": "Data quality validation",
        "module": "src.data_validation",
    },
]


def run_step(name, module):
    """Execute one project module and stop if it fails."""
    print("\n" + "=" * 70)
    print(f"RUNNING: {name}")
    print(f"MODULE:  {module}")
    print("=" * 70, flush=True)

    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        print(
            f"\nPIPELINE FAILED during: {name}",
            flush=True,
        )
        print(
            f"Exit code: {result.returncode}",
            flush=True,
        )
        return False

    print(f"\nCOMPLETED: {name}", flush=True)
    return True


def main():
    print("AI-POWERED DEMAND FORECASTING PIPELINE", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)

    for step in PIPELINE_STEPS:
        success = run_step(
            step["name"],
            step["module"],
        )

        if not success:
            raise SystemExit(1)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("Generated or refreshed outputs:")
    print("- data/processed/train_merged.csv")
    print("- data/processed/test_merged.csv")
    print("- data/processed/features_train.csv")
    print("- reports/data_quality_report.csv")


if __name__ == "__main__":
    main()
