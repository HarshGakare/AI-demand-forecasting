from pathlib import Path
import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
BUSINESS_DIR = REPORTS_DIR / "business"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_train.csv"

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI-Powered Demand Forecasting")
st.caption("Demand prediction, inventory recommendation and model performance analytics.")


@st.cache_data
def load_csv(path_string:str) -> pd.DataFrame:
    """Load a CSV file and cache it until refreshed."""
    
    path = Path(path_string)
    
    if not path.exists():
        return pd.DataFrame()
    
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()
    
    

def format_number(value, decimals: int = 2) -> str:
    """Format a number safely for dashboard display."""
    
    try:
        number = float(value)

        if pd.isna(number):
            return "N/A"
        
        return f"{number:,.{decimals}f}"
    
    except (TypeError, ValueError):
        return "N/A"
    

def find_column(dataframe: pd.DataFrame, candidates: list[str]) -> str | None :
    """Find a column without relying on capitalization."""
    
    lookup = {
        str(column).strip().lower(): column for column in dataframe.columns
    }
    
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
        
    return None

def display_missing_file(path: Path) -> None:
    st.info(f"Report not found: `{path.relative_to(PROJECT_ROOT)}`."
            "Run the corresponding project step to generate it.")
    

forecast_df = load_csv(str(REPORTS_DIR / "multi_horizon_predictions.csv"))
inventory_df = load_csv(str(REPORTS_DIR / "inventory_optimization.csv"))
model_metrics_df = load_csv(str(REPORTS_DIR / "lightgbm_metrics.csv"))
rolling_cv_df = load_csv(str(REPORTS_DIR / "rolling_cv_summary.csv"))
business_kpis_df = load_csv(str(BUSINESS_DIR / "executive_summary.csv"))
store_kpis_df = load_csv(str(BUSINESS_DIR / "store_business_kpis.csv"))
feature_sample_df = load_csv(str(FEATURES_PATH))

st.sidebar.header("Dashboard controls")

if st.sidebar.button("Refresh report data"):
    st.cache_data.clear()
    st.rerun()
    
st.sidebar.caption(
    "The dashboard reads your existing CSV reports."
    "Generate the reports before using their sections."
)

page = st.sidebar.radio("Choose a section",
                        ["Overview",
                         "Forecast Analysis",
                         "Inventory Optimization",
                         "Model Performance"])


if page == "Overview":
    st.header("Project Overview")
    
    forecast_rows = len(forecast_df)
    inventory_stores = (int(inventory_df["Store"].nunique()) if "Store" in inventory_df.columns else 0)
    
    col1,col2,col3,col4  = st.columns(4)
    
    col1.metric(
    "Forecast records",
    f"{forecast_rows:,}")

    col2.metric(
        "Stores in inventory report",
        f"{inventory_stores:,}")

    col3.metric(
        "Model metric reports",
        f"{len(model_metrics_df):,}",
    )

    col4.metric(
        "Business KPI records",
        f"{len(business_kpis_df):,}",
    )
    
    st.subheader("Available project outputs")
    
    available_reports = [
        REPORTS_DIR / "multi_horizon_predictions.csv",
        REPORTS_DIR / "multi_horizon_metrics.csv",
        REPORTS_DIR / "lightgbm_metrics.csv",
        REPORTS_DIR / "rolling_cv_summary.csv",
        REPORTS_DIR / "recursive_forecast_metrics.csv",
        REPORTS_DIR / "inventory_optimization.csv",
        BUSINESS_DIR / "executive_summary.csv",
        BUSINESS_DIR / "business_recommendations.csv",
    ]
    
    
    report_status = pd.DataFrame([{
        "Report": str(path.relative_to(PROJECT_ROOT)),
        "Available": path.exists()
    } for path in available_reports ])
    
    
    st.dataframe(report_status,use_container_width=True, hide_index=True)
    
    if not business_kpis_df.empty:
        st.subheader("Executive summary")
        st.dataframe(
            business_kpis_df,
            use_container_width=True,
            hide_index=True
        )
        
    
    st.warning(
        "Inventory quantities currently use a proxy for available "
        "inventory, not actual stock-on-hand data. Treat reorder "
        "recommendations as illustrative until real inventory data "
        "is connected."
    )
    
    
elif page == "Demand Prediction":
    st.header("Predict Store Demand")
    
    st.write("This page calls your FastAPI service. The current endpoint only predicts dates with existing engineered features rows.")
    
    if feature_sample_df.empty:
        st.error("Feature data is unavailable. Generated `data/processed/features_train.csv` first.")
        
    elif "Store" not in feature_sample_df.columns:
        st.error("The features dataset does not contain a Store column.")
        
    else:
        stores = sorted(pd.to_numeric(feature_sample_df["Store"], errors="coerce")).dropna().astype(int).unique().tolist()
        
        
        selected_store = st.selectbox("Select store", stores)
        available_dates= feature_sample_df.loc[feature_sample_df["Store"] == selected_store, "Date"]
        available_dates = pd.to_datetime(available_dates, errors="coerce").dropna()
        
        
        if available_dates.empty:
            st.warning("No dates are available for the selected store")
            
        else:
            min_date = available_dates.min().date()
            max_date = available_dates.max().date()
            
            selected_date = st.date_input(
                "Select prediction date",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
            
            
            if st.button("Predict slaes", type="primary"):
                payload = {"store": int(selected_store),
                           "date": selected_date.isoformat()}
                
                try:
                    with st.spinner("Requesting prediction from the API..."):
                        response = requests.post(f"{API_URL}/predict",
                                                 json=payload,
                                                 timeout=30)
                        
                    if response.ok:
                        result = response.json()
                        
                        st.success("Prediction received successfully")
                        
                        result_col1, result_col2 = st.columns(2)
                        result_col2.metric("Store", str(result.get("store", selected_store)))
                        
                        st.write(f"**Date:** {result.get('date', selected_date)}")
                        st.write(f"**Model:** {result.get('model','Unknown')}")
                    
                    else:
                        try:
                            detail = response.json().get("detail", response.text)
                        
                        except ValueError:
                            detail = response.text
                            
                        st.error(f"API returned HTTP {response.status_code}: {detail}")
                
                except requests.RequestException:
                    st.error("Could not connect to the API. Start it in a separate terminal with: \n\n `python -m uvicorn src.api:app --reload`")
                    
        
    st.info(
        "For this version, select a date inside the available feature dataset. Forecasting arbitrary future dates will require generating future features and handling lag values."
    )
    
    
    
elif page == "Forecast Analysis":
    st.header("Forecast Analysis")
    
    if forecast_df.empty:
        display_missing_file(REPORTS_DIR / "multi_horizon_predictions.csv")
    
    else:
        store_col = find_column(forecast_df,["Store"])
        date_col = find_column(forecast_df,["Date"])
        horizon_col = find_column(forecast_df, ["Horizon"])
        actual_col = find_column(forecast_df, ["Sales","ActualSales","Actual"])
        prediction_col = find_column(forecast_df,["Prediction","PredictedSales","Forecast"])
        filtered_df = forecast_df.copy()
        
        if store_col:
            store_values = sorted(pd.to_numeric(filtered_df[store_col],errors="coerce").dropna().astype(int).unique().tolist() )

            selected_stores = st.multiselect(
                "Filter stores",
                store_values,
                default=store_values[: min(10, len(store_values))],
            )

            filtered_df = filtered_df.loc[pd.to_numeric(
                    filtered_df[store_col],
                    errors="coerce",
                ).isin(selected_stores)
            ]

        if horizon_col:
            horizons = sorted(
                pd.to_numeric(
                    filtered_df[horizon_col],
                    errors="coerce",
                ).dropna().unique().tolist()
            )

            if horizons:
                selected_horizon = st.selectbox(
                    "Forecast horizon (days)",
                    horizons,
                    index=len(horizons) - 1,
                )

                filtered_df = filtered_df.loc[
                    pd.to_numeric(
                        filtered_df[horizon_col],
                        errors="coerce",
                    ) == selected_horizon
                ]

        if date_col:
            filtered_df[date_col] = pd.to_datetime(
                filtered_df[date_col],
                errors="coerce",
            )

            chart_col1, chart_col2 = st.columns(2)

            if prediction_col:
                chart_data = filtered_df.dropna(
                    subset=[date_col, prediction_col]
                ).copy()

                if not chart_data.empty:
                    chart_data[prediction_col] = pd.to_numeric(
                        chart_data[prediction_col],
                        errors="coerce",
                    )

                    prediction_by_date = (
                        chart_data.groupby(date_col)[prediction_col]
                        .mean()
                        .rename("Average predicted sales")
                    )

                    with chart_col1:
                        st.subheader("Average predicted sales")
                        st.line_chart(prediction_by_date)

            if actual_col and prediction_col:
                comparison_df = filtered_df.copy()

                comparison_df[actual_col] = pd.to_numeric(
                    comparison_df[actual_col],
                    errors="coerce",
                )
                comparison_df[prediction_col] = pd.to_numeric(
                    comparison_df[prediction_col],
                    errors="coerce",
                )

                comparison_df = comparison_df.dropna(
                    subset=[date_col, actual_col, prediction_col]
                )

                if not comparison_df.empty:
                    daily_comparison = comparison_df.groupby(date_col)[[actual_col, prediction_col]].mean()

                    daily_comparison.columns = [
                        "Actual sales",
                        "Predicted sales",
                    ]

                    with chart_col2:
                        st.subheader("Actual vs predicted")
                        st.line_chart(daily_comparison)

        st.subheader("Forecast records")

        st.dataframe(
            filtered_df.head(500),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Charts aggregate rows by date. Select a single horizon when comparing predictions across dates."
        )


elif page == "Inventory Optimization":
    st.header("Inventory Optimization")

    if inventory_df.empty:
        display_missing_file(REPORTS_DIR / "inventory_optimization.csv")

    else:
        numeric_columns = [
            column
            for column in inventory_df.columns
            if pd.api.types.is_numeric_dtype(inventory_df[column])
        ]

        status_col = find_column(
            inventory_df,
            [
                "InventoryStatus",
                "ReorderStatus",
                "Status",
            ],
        )

        if status_col:
            st.subheader("Inventory status")

            status_counts = (
                inventory_df[status_col]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
            )

            st.bar_chart(status_counts)

        preferred_columns = [
            "Store",
            "AverageDailyDemand",
            "ForecastDailyDemand",
            "SafetyStock",
            "ReorderPoint",
            "TargetInventory",
            "CurrentInventoryProxy",
            "RecommendedOrderQuantity",
        ]

        visible_columns = [
            column
            for column in preferred_columns
            if column in inventory_df.columns
        ]

        if not visible_columns:
            visible_columns = inventory_df.columns.tolist()

        display_df = inventory_df[visible_columns].copy()

        if "RecommendedOrderQuantity" in display_df.columns:
            display_df["RecommendedOrderQuantity"] = pd.to_numeric(
                display_df["RecommendedOrderQuantity"],
                errors="coerce",
            )

            display_df = display_df.sort_values(
                "RecommendedOrderQuantity",
                ascending=False,
            )

        st.subheader("Store-level recommendations")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download inventory report",
            data=inventory_df.to_csv(index=False).encode("utf-8"),
            file_name="inventory_optimization.csv",
            mime="text/csv",
        )

        st.warning(
            "The current inventory value is a proxy derived from historical demand. It is not a measured on-hand stock quantity. These recommendations are for demonstration and should not be used directly for purchasing."
        )



elif page == "Model Performance":
    st.header("Model Performance")

    if not model_metrics_df.empty:
        st.subheader("Final holdout metrics")

        st.dataframe(
            model_metrics_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        display_missing_file(
            REPORTS_DIR / "lightgbm_metrics.csv"
        )

    if not rolling_cv_df.empty:
        st.subheader("Rolling cross-validation")

        st.dataframe(
            rolling_cv_df,
            use_container_width=True,
            hide_index=True,
        )

        numeric_cv_columns = [
            column
            for column in rolling_cv_df.columns
            if pd.api.types.is_numeric_dtype(rolling_cv_df[column])
        ]

        metric_candidates = [
            column
            for column in numeric_cv_columns
            if any(
                metric in column.lower()
                for metric in ["mae", "rmse", "wape"]
            )
        ]

        if metric_candidates:
            st.subheader("Cross-validation metric summary")
            st.bar_chart(
                rolling_cv_df[metric_candidates].iloc[0]
            )

    else:
        display_missing_file(
            REPORTS_DIR / "rolling_cv_summary.csv"
        )

    st.caption(
        "MAE and RMSE are measured in sales units. WAPE is a percentage; lower values generally indicate better forecasts."
    )




st.divider()

st.caption(
    "AI-Powered Demand Forecasting and Inventory Optimization | "
    "Python, LightGBM, FastAPI, Streamlit"
)
    
        
        
        
        
        