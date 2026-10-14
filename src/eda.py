from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_merged.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

def load_data(): 
    """Load merged historial sales data."""
    
    if not DATA_PATH.exists():
        raise FileExistsError("Processed data not found. Run data_loader.py first")
    
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    
    return df

def prepare_data(df):
    """Create analysis only calender fields."""
    
    df = df.copy()
    
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Weekday"] = df["Date"].dt.day_name()
    
    return df

def save_plot(filename):
    """Save the current Matplotlib Figure"""
    
    plt.tight_layout()
    
    plt.savefig(
        FIGURE_DIR / filename,
        dpi=150,
        bbox_inches='tight'
    )
    
    plt.close()
    
def plot_total_sales(df):
    """Plot total daily sales across all stores."""
    
    daily_sales = (
        df.groupby("Date", as_index=False)["Sales"]
        .sum()
    )
    
    plt.figure(figsize=(14, 5))
    
    sns.lineplot(
        data=daily_sales,
        x="Date",
        y="Sales",
    )
    
    plt.title("Total Daily Sales Across All Stores")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    save_plot("total_daily_sales.png")
    
def plot_weekday_sales(df):
    """Compare sales by weekday"""
    
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    
    weekday_sales = (
        df.groupby("Weekday", as_index=False)["Sales"].mean()
    )
    
    plt.figure(figsize=(11,5))
    
    sns.barplot(
        data=weekday_sales,
        x="Weekday",
        y="Sales",
        order=weekday_order
    )
    
    plt.title("AVG salese by weekday")
    plt.xlabel("Weekday")
    plt.ylabel("AVG Sales")
    plt.xticks(rotation=30)
    save_plot("weekday_sales.png")
    

def plot_promotion_effect(df):
    """Compare AVG sales with and without promotions."""
    
    promotion_sales = (
        df.groupby("Promo", as_index=False)["Sales"].mean()
    )
    
    promotion_sales["Promo"] = promotion_sales["Promo"].map({
        0:"No Promotion",
        1:"Promotion",
    })
    
    plt.figure(figsize=(8,5))

    sns.barplot( data=promotion_sales,
                x="Promo",
                y="Sales")
    
    plt.title("AVG sales: Promotion VS No Promotion")
    plt.xlabel("Promotion Status")
    plt.ylabel("AVG Sales")
    
    save_plot("promotion_effect.png")
    
def plot_store_type_sales(df):
    """Compare AVG sales by store type"""
    
    store_sales = (
        df.groupby("StoreType", as_index=False)["Sales"]
        .mean()
        .sort_values("Sales", ascending=False)
    )
    
    plt.figure(figsize=(8,5))
    
    sns.barplot(
        data=store_sales,
        x="StoreType",
        y="Sales"
    )
    plt.title("AVG Sales by Store Type")
    plt.xlabel("Store Type")
    plt.ylabel("AVG Sales")
    
    save_plot("store_type_sales.png")
    

def plot_monthly_sales(df):
    """Plot monthly sales across all stores."""
    
    monthly_sales = (
        df.groupby(["Year","Month"], as_index=False)["Sales"].sum()
    )
    
    monthly_sales["YearMonth"] = (
        monthly_sales["Year"].astype(str)
        + "-"
        + monthly_sales["Month"].astype(str).str.zfill(2)
    )
    
    
    plt.figure(figsize=(14,5))
    
    sns.lineplot(
        data=monthly_sales,
        x="YearMonth",
        y="Sales",
        markers="o",
    )

    plt.title("Monthly Sales Trend")
    plt.xlabel("Year-Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=60)

    save_plot("monthly_sales.png")
    
def plot_sales_distribution(df):
    """Inspect the distribution of daily slaes"""
    
    plt.figure(figsize=(10,5))
    
    sns.histplot(
        data=df,
        x="Sales",
        bins=60,
        kde=True
    )
    
    plt.title("Daily Sales Distribution")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    
    save_plot("sales_distribution.png")
    

def print_summary(df):
    """Print important descriptive statistics"""
    
    print("\nSales summary:")
    print(df["Sales"].describe())
    
    print("\nStores:")
    print(df["Store"].nunique())
    
    print("\nData range:")
    print(df["Date"].min(), "to", df["Date"].max())
    
    print("\nClosed-store rows:")
    if "Open" in df.columns:
        print((df["Open"] == 0).sum())
        
    print("\nZero-sales rows:")
    print((df["Sales"] == 0).sum())
    
    print("\nPromotion distribution:")
    print(df["Promo"].value_counts(dropna=False))
    

def main():
    df = load_data()
    df = prepare_data(df)
    
    print_summary(df)
    
    plot_total_sales(df)
    plot_weekday_sales(df)
    plot_promotion_effect(df)
    plot_store_type_sales(df)
    plot_monthly_sales(df)
    plot_sales_distribution(df)
    
    print(f"\nFigure save to : {FIGURE_DIR}")
        
if __name__ == "__main__":
    main()
        