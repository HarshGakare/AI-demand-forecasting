from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (PROJECT_ROOT/ "data"/ "processed"/ "features_train.csv")
SALES_PATH = ( PROJECT_ROOT / "data" / "processed" / "train_merged.csv")

FORECAST_PATH = ( PROJECT_ROOT / "reports" / "multi_horizon_predictions.csv")
OUTPUT_DIR = ( PROJECT_ROOT / "reports")

OUTPUT_PATH = (OUTPUT_DIR / "inventory_optimization.csv" )
SUMMARY_PATH = ( OUTPUT_DIR / "inventory_summary.csv")



LEAD_TIME_DAYS = 7
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL = 0.95
Z_SCORE = 1.645
MIN_STD = 1.0



def load_data():
    """
    Load historical sales, engineered features and LightGBM forecasts.
    """

    print("Loading historical sales...")

    sales_df = pd.read_csv( SALES_PATH, parse_dates=["Date"], low_memory=False )

    print("Loading forecasts...")

    forecast_df = pd.read_csv( FORECAST_PATH, parse_dates=["Date","ForecastOrigin"])

    print("Loading feature data...")

    features_df = pd.read_csv( FEATURES_PATH, parse_dates=["Date"], low_memory=False)

    return (sales_df,forecast_df,features_df )


# VALIDATE FORECAST FILE

def validate_forecast_columns(forecast_df):
    """
    Validate the expected multi-horizon forecast format.
    """

    required_columns = { "Store", "Date", "Sales", "Prediction", "Horizon"}

    missing = (required_columns - set(forecast_df.columns))

    if missing:
        raise ValueError(
            f"Forecast file is missing columns: {sorted(missing)}"
        )


# SELECT 30-DAY FORECAST

def select_forecast_horizon(forecast_df,horizon=30):
    """
    Select the requested forecasting horizon.

    The 30-day LightGBM forecast is used as the demand planning horizon for inventory optimization.
    """

    selected = forecast_df[ forecast_df["Horizon"] == horizon].copy()

    if selected.empty:

        raise ValueError(
            f"No forecast rows found for {horizon}-day horizon."
        )

    return selected


# HISTORICAL DEMAND STATISTICS

def calculate_demand_statistics(sales_df, forecast_origin ):
    """
    Calculate store-level historical demand statistics using data available before the forecast origin.
    A recent 56-day window is used to represent current demand variability.
    """

    history = sales_df[sales_df["Date"] <= forecast_origin].copy()

    recent_start = ( forecast_origin - pd.Timedelta(days=56))

    history = history[ history["Date"] > recent_start].copy()

    statistics = ( history.groupby("Store")["Sales"]
        .agg(
            HistoricalMeanDemand="mean",
            HistoricalStdDemand="std",
            HistoricalMinDemand="min",
            HistoricalMaxDemand="max",
        ).reset_index()
    )

    statistics["HistoricalStdDemand"] = statistics["HistoricalStdDemand"].fillna(MIN_STD)
    statistics["HistoricalStdDemand"] = statistics["HistoricalStdDemand"].clip(lower=MIN_STD)

    return statistics


# FORECAST AGGREGATION

def aggregate_forecast_demand(forecast_df):
    """
    Aggregate daily forecast into store-level planning quantities.
    """

    forecast_summary = (forecast_df.groupby("Store")
        .agg(
            ForecastDemand=("Prediction","sum",),
            AverageDailyForecast=( "Prediction","mean",),
            PeakDailyForecast=( "Prediction","max" ),
        ).reset_index()
    )

    return forecast_summary


# INVENTORY CALCULATIONS

def calculate_inventory_metrics( forecast_summary, demand_statistics):
    """
    Calculate safety stock, reorder point and target inventory for each store.
    """

    result = forecast_summary.merge(demand_statistics, on="Store", how="left" )


    result["HistoricalMeanDemand"] = result["HistoricalMeanDemand"].fillna(result["AverageDailyForecast"] )
    result[ "HistoricalStdDemand"] = result[ "HistoricalStdDemand"].fillna(MIN_STD)
    result["HistoricalStdDemand"] = result["HistoricalStdDemand"].clip(lower=MIN_STD)

    # Demand during lead time
    result["LeadTimeDemand"] = ( result["AverageDailyForecast"] * LEAD_TIME_DAYS )

    # Safety Stock = Z × demand standard deviation × sqrt(lead time)
    result[ "SafetyStock"] = (Z_SCORE * result["HistoricalStdDemand"] * np.sqrt(LEAD_TIME_DAYS))

    # Reorder point = Lead-time demand + safety stock
    result["ReorderPoint"] = (result["LeadTimeDemand"] + result["SafetyStock"])

    # Target inventory Cover lead time + review period.
    target_days = (LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS)

    result["TargetInventory" ] = ( result["AverageDailyForecast"] * target_days + 
                                  result["SafetyStock"])

    # Current inventory proxy
    # We do not have actual on-hand inventory in Rossmann,
    # so use the most recent observed sales level as a planning proxy.

    result["CurrentInventoryProxy"] = result["HistoricalMeanDemand"] * LEAD_TIME_DAYS

    # Recommended order quantity
    # Order enough to reach target inventory.

    result["RecommendedOrderQuantity"] = ( result["TargetInventory"] - 
                                          result["CurrentInventoryProxy"]).clip(lower=0)

    # Inventory status

    result["InventoryStatus"] = np.where(result["CurrentInventoryProxy"] <= 
                                         result["ReorderPoint"],"REORDER","SUFFICIENT", )

    return result


# RISK CLASSIFICATION

def classify_inventory_risk( result):
    """
    Classify stores by forecast/inventory risk.
    """

    result = result.copy()

    result["DemandVariabilityRatio"] = (result["HistoricalStdDemand"]
                                        / result["AverageDailyForecast"].replace(0, np.nan) )

    result[ "DemandVariabilityRatio"] = result["DemandVariabilityRatio"].fillna(0)

    result["RiskLevel" ] = np.select( [ result["DemandVariabilityRatio"] >= 0.75,
                                       result["DemandVariabilityRatio"] >= 0.40,],
                                     [ "HIGH", "MEDIUM",],default="LOW")

    return result


# SUMMARY

def create_summary(inventory_df):
    """
    Create management-level inventory summary.
    """

    total_stores = len(inventory_df )
    reorder_stores = int((inventory_df["InventoryStatus" ]  == "REORDER").sum())
    high_risk_stores = int((inventory_df["RiskLevel"] == "HIGH").sum())

    total_forecast_demand = (inventory_df["ForecastDemand"].sum())
    total_safety_stock = (inventory_df["SafetyStock"].sum() )
    total_recommended_order = (inventory_df["RecommendedOrderQuantity"].sum())

    average_service_level = (SERVICE_LEVEL * 100)

    summary = pd.DataFrame([{
                "TotalStores": total_stores,
                "StoresRequiringReorder": reorder_stores,
                "StoresReorderPercentage": (reorder_stores / total_stores * 100 ),
                "HighRiskStores": high_risk_stores,
                "HighRiskPercentage": ( high_risk_stores / total_stores * 100),
                "Total30DayForecastDemand": ( total_forecast_demand ),
                "TotalSafetyStock": ( total_safety_stock),
                "TotalRecommendedOrderQuantity": ( total_recommended_order ),
                "AssumedServiceLevel": ( average_service_level ),
                "LeadTimeDays": ( LEAD_TIME_DAYS ),
                "ReviewPeriodDays": ( REVIEW_PERIOD_DAYS ),
            }])

    return summary


# MAIN

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("INVENTORY OPTIMIZATION")
    print("=" * 60)

    (sales_df,forecast_df,features_df) = load_data()

    print( f"Historical sales rows: {len(sales_df):,}")
    print(f"Forecast rows: {len(forecast_df):,}")


    validate_forecast_columns(forecast_df)

    planning_forecast = (select_forecast_horizon(forecast_df, horizon=30 ))

    forecast_origin = ( planning_forecast["ForecastOrigin"].iloc[0])

    print(f"Forecast origin: {forecast_origin.date()}")
    print(f"Planning horizon: 30 days")
    print( f"Stores: {planning_forecast['Store'].nunique():,}" )


    forecast_summary = (aggregate_forecast_demand( planning_forecast ))
    demand_statistics = (calculate_demand_statistics(sales_df, forecast_origin ))
    inventory_df = ( calculate_inventory_metrics( forecast_summary, demand_statistics))
    inventory_df = (classify_inventory_risk(inventory_df))

    # Round numeric outputs

    numeric_columns = [
        "ForecastDemand",
        "AverageDailyForecast",
        "PeakDailyForecast",
        "HistoricalMeanDemand",
        "HistoricalStdDemand",
        "LeadTimeDemand",
        "SafetyStock",
        "ReorderPoint",
        "TargetInventory",
        "CurrentInventoryProxy",
        "RecommendedOrderQuantity",
        "DemandVariabilityRatio",
    ]

    for column in numeric_columns:
        if column in inventory_df.columns:
            inventory_df[column] = ( inventory_df[column] .round(2) )


    inventory_df.to_csv(OUTPUT_PATH,index=False)

    print()
    print( f"Saved inventory optimization \n{OUTPUT_PATH}")


    summary = create_summary(inventory_df)

    summary.to_csv(SUMMARY_PATH,index=False)

    print(f"Saved inventory summary: \n{SUMMARY_PATH}")

    # Display

    print()
    print("=" * 60)
    print("INVENTORY SUMMARY")
    print("=" * 60)

    row = summary.iloc[0]

    print(
        f"Total stores: "
        f"{int(row['TotalStores']):,}"
    )

    print(
        f"Stores requiring reorder: "
        f"{int(row['StoresRequiringReorder']):,}"
    )

    print(
        f"Reorder percentage: "
        f"{row['StoresReorderPercentage']:.2f}%"
    )

    print(
        f"High-risk stores: "
        f"{int(row['HighRiskStores']):,}"
    )

    print(
        f"30-day forecast demand: "
        f"{row['Total30DayForecastDemand']:,.2f}"
    )

    print(
        f"Total safety stock: "
        f"{row['TotalSafetyStock']:,.2f}"
    )

    print(
        f"Recommended order quantity: "
        f"{row['TotalRecommendedOrderQuantity']:,.2f}"
    )

    print(
        f"Assumed service level: "
        f"{row['AssumedServiceLevel']:.0f}%"
    )

    print(
        f"Lead time: "
        f"{int(row['LeadTimeDays'])} days"
    )

    print(
        f"Review period: "
        f"{int(row['ReviewPeriodDays'])} days"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()