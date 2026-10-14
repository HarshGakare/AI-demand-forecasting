
from pathlib import Path

import numpy as np
import pandas as pd




PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"

INVENTORY_PATH = (REPORT_DIR / "inventory_optimization.csv")
INVENTORY_SUMMARY_PATH = ( REPORT_DIR / "inventory_summary.csv")

FORECAST_PATH = (REPORT_DIR / "multi_horizon_predictions.csv")
ROLLING_CV_PATH = (REPORT_DIR / "rolling_cv_summary.csv")

OUTPUT_DIR = REPORT_DIR / "business"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def load_inputs():
    """Load existing model and inventory reports."""

    required_files = [INVENTORY_PATH,FORECAST_PATH ,ROLLING_CV_PATH ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    inventory = pd.read_csv(INVENTORY_PATH)
    forecasts = pd.read_csv(FORECAST_PATH,parse_dates=["Date"])
    cv = pd.read_csv(ROLLING_CV_PATH)

    required_inventory_columns = {
        "Store",
        "ForecastDemand",
        "AverageDailyForecast",
        "HistoricalMeanDemand",
        "HistoricalStdDemand",
        "SafetyStock",
        "ReorderPoint",
        "TargetInventory",
        "RecommendedOrderQuantity",
        "InventoryStatus",
        "RiskLevel",
    }

    missing = (required_inventory_columns - set(inventory.columns))

    if missing:
        raise ValueError(
            f"Inventory report is missing columns: {sorted(missing)}"
        )

    return inventory, forecasts, cv




def validate_data(inventory, forecasts):
    """Check basic integrity before calculating KPIs."""

    if inventory["Store"].duplicated().any():
        raise ValueError("Inventory report contains duplicate stores.")

    if inventory["Store"].isna().any():
        raise ValueError("Inventory report contains missing store IDs." )

    if not np.isfinite(
        inventory["ForecastDemand"].dropna()).all():
        raise ValueError("Forecast demand contains infinite values.")

    if (inventory["ForecastDemand"].dropna() < 0).any():
        raise ValueError("Forecast demand contains negative values.")

    if forecasts.empty:
        raise ValueError("Forecast dataset is empty.")



def create_store_kpis(inventory):
    """Calculate additional store-level business KPIs."""

    df = inventory.copy()

    df["ForecastDailyDemand"] = ( df["ForecastDemand"] / 30)

    df["DemandVolatilityRatio"] = (
        df["HistoricalStdDemand"] / df["HistoricalMeanDemand"].replace(0, np.nan)
    )

    df["DemandVolatilityRatio"] = (df["DemandVolatilityRatio"].fillna(0) )

    df["SafetyStockCoverageDays"] = (
        df["SafetyStock"] / df["AverageDailyForecast"].replace(0, np.nan)
    ).fillna(0)

    df["TargetInventoryCoverageDays"] = (
        df["TargetInventory"] / df["AverageDailyForecast"].replace(0, np.nan)
    ).fillna(0)

    df["RecommendedOrderToDemandRatio"] = (
        df["RecommendedOrderQuantity"] / df["ForecastDemand"].replace(0, np.nan)
    ).fillna(0)

    # Transparent priority score for review, not a learned model.
    
    df["BusinessPriorityScore"] = (
        0.50 * df["RiskLevel"].map({"HIGH": 3, "MEDIUM": 2, "LOW": 1}).fillna(1)
        + 0.30 * df["InventoryStatus"].map( {"REORDER": 2, "SUFFICIENT": 0} ).fillna(0)
        + 0.20 * (df["DemandVolatilityRatio"].rank(pct=True) * 3 )
    )

    df["BusinessPriority"] = np.select( [ df["BusinessPriorityScore"] >= 2.5, df["BusinessPriorityScore"] >= 1.8], [ "HIGH","MEDIUM",], default="LOW" )

    return df




def create_forecast_kpis(forecasts):
    """Summarize forecast volume and horizon coverage."""

    df = forecasts.copy()

    if "Horizon" in df.columns:
        df["Horizon"] = pd.to_numeric(df["Horizon"], errors="coerce")

    rows = []

    if "Horizon" in df.columns:
        horizons = sorted(df["Horizon"].dropna().unique())
    else:
        horizons = [None]


    for horizon in horizons:
        subset = (df[df["Horizon"] == horizon]
            if horizon is not None
            else df
        )

        if subset.empty:
            continue

        rows.append(
            {
                "HorizonDays": horizon,
                "Rows": len(subset),
                "Stores": subset["Store"].nunique(),
                "ForecastUnits": subset["Prediction"].sum(),
                "MeanDailyPrediction": subset["Prediction"].mean(),
                "MedianDailyPrediction": subset["Prediction"].median(),
                "ZeroPredictionRows": int((subset["Prediction"] == 0).sum())
            }
        )

    return pd.DataFrame(rows)



def create_model_kpis(cv):
    """
    Preserve the rolling-CV results as model performance KPIs.
    """

    numeric_columns = [
        column
        for column in cv.columns
        if column.lower().endswith("_mean")
        or column.lower().endswith("_std")
        or column.lower().endswith("_median")
    ]

    if not numeric_columns:
        # Keep source column names visible instead of silently inventing metric values.
        return cv.copy()

    return cv[numeric_columns].copy()



def create_executive_summary(store_kpis, forecast_kpis):
    """Create concise management-level KPIs."""

    store_count = store_kpis["Store"].nunique()
    high_risk = int( (store_kpis["RiskLevel"] == "HIGH").sum())
    reorder_count = int( (store_kpis["InventoryStatus"] == "REORDER").sum())
    high_priority = int((store_kpis["BusinessPriority"] == "HIGH").sum())

    summary = pd.DataFrame(
        [
            {
                "StoresAnalyzed": store_count,
                "StoresFlaggedForReorder": reorder_count,
                "ReorderFlagPercentage": (
                    reorder_count / store_count * 100
                    if store_count else np.nan
                ),
                "HighRiskStores": high_risk,
                "HighRiskPercentage": (
                    high_risk / store_count * 100
                    if store_count else np.nan
                ),
                "HighPriorityStores": high_priority,
                "TotalForecastDemandUnits": (
                    store_kpis["ForecastDemand"].sum()
                ),
                "TotalSafetyStockUnits": (
                    store_kpis["SafetyStock"].sum()
                ),
                "TotalRecommendedOrderUnits": (
                    store_kpis["RecommendedOrderQuantity"].sum()
                ),
                "AverageDemandVolatilityRatio": (
                    store_kpis["DemandVolatilityRatio"].mean()
                ),
                "ForecastRowsAvailable": (
                    int(forecast_kpis["Rows"].sum())
                    if not forecast_kpis.empty else 0
                ),
            }
        ]
    )

    return summary




def create_recommendations(store_kpis):
    """Generate transparent, rule-based action suggestions."""

    recommendations = store_kpis[
        [
            "Store",
            "BusinessPriority",
            "BusinessPriorityScore",
            "InventoryStatus",
            "RiskLevel",
            "ForecastDemand",
            "SafetyStock",
            "ReorderPoint",
            "RecommendedOrderQuantity",
        ]
    ].copy()

    recommendations["RecommendedAction"] = np.select(
        [ ( recommendations["BusinessPriority"] == "HIGH") & ( recommendations["InventoryStatus"] == "REORDER" ),
            recommendations["InventoryStatus"] == "REORDER",
            recommendations["RiskLevel"] == "HIGH",
        ],
        [
            "Review stock position and replenishment urgently",
            "Review replenishment requirement",
            "Review demand variability and safety-stock assumptions",
        ],
        default="Monitor forecast and inventory position",
    )

    recommendations = recommendations.sort_values(
        [ "BusinessPriorityScore","ForecastDemand"],
        ascending=[False, False],
    )

    return recommendations




def main():
    print("=" * 65)
    print("BUSINESS KPI ANALYSIS")
    print("=" * 65)

    inventory, forecasts, cv = load_inputs()

    validate_data(inventory, forecasts)

    store_kpis = create_store_kpis(inventory)
    forecast_kpis = create_forecast_kpis(forecasts)
    model_kpis = create_model_kpis(cv)

    executive_summary = create_executive_summary(store_kpis, forecast_kpis)

    recommendations = create_recommendations( store_kpis )

    # Save detailed reports.
    store_kpis.to_csv( OUTPUT_DIR / "store_business_kpis.csv", index=False )
    forecast_kpis.to_csv( OUTPUT_DIR / "forecast_business_kpis.csv", index=False )
    model_kpis.to_csv( OUTPUT_DIR / "model_performance_kpis.csv", index=False )
    executive_summary.to_csv( OUTPUT_DIR / "executive_summary.csv", index=False )
    recommendations.to_csv(OUTPUT_DIR / "business_recommendations.csv",index=False)

    print("\nEXECUTIVE SUMMARY")
    print("-" * 65)
    print(executive_summary.to_string(index=False))

    print("\nTOP 10 STORES FOR BUSINESS REVIEW")
    print("-" * 65)

    print(
        recommendations.head(10).to_string(index=False)
    )

    print("\nReports saved to:")
    print(OUTPUT_DIR)

    print("\nImportant interpretation:")
    print(
        "Inventory recommendations are estimates. "
        "Actual on-hand stock, supplier lead times, "
        "and order constraints are not available in "
        "the Rossmann dataset."
    )


if __name__ == "__main__":
    main()
