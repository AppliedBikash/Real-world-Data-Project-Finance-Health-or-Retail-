import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")


def generate_or_load_dataset(filepath=os.path.join(DATA_DIR, "retail_sales_data.csv"), n_samples=2500):
    """Generates a realistic real-world Retail & E-Commerce Sales dataset if not already present."""
    if os.path.exists(filepath):
        print(f"Loading existing dataset from {filepath}...")
        return pd.read_csv(filepath, parse_dates=["Order_Date"])

    print(f"Generating realistic real-world retail dataset ({n_samples} records)...")
    np.random.seed(42)

    dates = pd.date_range(start="2025-01-01", end="2025-12-31", periods=n_samples)
    categories = ["Electronics", "Home & Kitchen", "Apparel", "Health & Beauty", "Sports & Outdoors"]
    cat_probs = [0.28, 0.24, 0.20, 0.16, 0.12]
    regions = ["North America", "Europe", "Asia-Pacific", "Latin America"]
    reg_probs = [0.40, 0.30, 0.20, 0.10]
    segments = ["Consumer", "Corporate", "VIP Member"]
    seg_probs = [0.55, 0.30, 0.15]
    channels = ["Web Store", "Mobile App", "Marketplace"]
    chan_probs = [0.50, 0.35, 0.15]

    cat_choice = np.random.choice(categories, size=n_samples, p=cat_probs)
    reg_choice = np.random.choice(regions, size=n_samples, p=reg_probs)
    seg_choice = np.random.choice(segments, size=n_samples, p=seg_probs)
    chan_choice = np.random.choice(channels, size=n_samples, p=chan_probs)

    base_price_map = {
        "Electronics": 220.0,
        "Home & Kitchen": 95.0,
        "Sports & Outdoors": 115.0,
        "Apparel": 60.0,
        "Health & Beauty": 45.0,
    }
    margin_map = {
        "Electronics": 0.28,
        "Home & Kitchen": 0.38,
        "Sports & Outdoors": 0.34,
        "Apparel": 0.48,
        "Health & Beauty": 0.52,
    }

    unit_prices = np.array([
        np.round(np.random.gamma(shape=4.0, scale=base_price_map[c] / 4.0), 2)
        for c in cat_choice
    ])
    unit_prices = np.clip(unit_prices, 12.0, 1200.0)

    quantities = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.35, 0.30, 0.18, 0.10, 0.05, 0.02])
    # VIP & Corporate order slightly higher quantities
    quantities = np.where(seg_choice == "VIP Member", quantities + 1, quantities)
    quantities = np.where(seg_choice == "Corporate", quantities + np.random.choice([0, 1, 2], size=n_samples), quantities)

    discount_rate = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], size=n_samples,
                                     p=[0.35, 0.20, 0.18, 0.12, 0.08, 0.04, 0.03])

    marketing_spend = np.round(np.random.uniform(2.0, 45.0, size=n_samples) * (1 + 0.3 * (cat_choice == "Electronics")), 2)
    customer_tenure_months = np.random.randint(1, 60, size=n_samples)
    customer_rating = np.round(np.clip(np.random.normal(4.3, 0.45, size=n_samples) - 0.4 * (discount_rate > 0.20), 2.5, 5.0), 1)

    # Seasonal Q4 multiplier
    months = pd.Series(dates).dt.month.values
    seasonal_boost = np.where(np.isin(months, [11, 12]), 1.18, 1.0)

    gross_sales = unit_prices * quantities * seasonal_boost
    net_revenue = np.round(gross_sales * (1.0 - discount_rate) + 1.8 * marketing_spend + np.random.normal(0, 8, size=n_samples), 2)
    net_revenue = np.clip(net_revenue, 10.0, None)

    base_margins = np.array([margin_map[c] for c in cat_choice])
    effective_margin = base_margins - (0.65 * discount_rate)
    profit = np.round(net_revenue * effective_margin - (0.35 * marketing_spend), 2)

    df = pd.DataFrame({
        "Order_ID": [f"ORD-{10000 + i}" for i in range(n_samples)],
        "Order_Date": dates,
        "Category": cat_choice,
        "Region": reg_choice,
        "Customer_Segment": seg_choice,
        "Sales_Channel": chan_choice,
        "Unit_Price": unit_prices,
        "Quantity": quantities,
        "Discount_Rate": discount_rate,
        "Marketing_Spend": marketing_spend,
        "Customer_Tenure_Months": customer_tenure_months,
        "Customer_Rating": customer_rating,
        "Revenue": net_revenue,
        "Profit": profit,
    })

    df.to_csv(filepath, index=False)
    print(f"Saved dataset to {filepath}")
    return df


def perform_exploratory_analysis(df):
    """Performs statistical summarization and generates publication-ready EDA visualizations."""
    print("\n" + "=" * 65)
    print("1. EXPLORATORY DATA ANALYSIS & DOMAIN SUMMARY")
    print("=" * 65)

    cat_summary = df.groupby("Category").agg(
        Orders=("Order_ID", "count"),
        Total_Revenue=("Revenue", "sum"),
        Mean_Revenue=("Revenue", "mean"),
        Median_Revenue=("Revenue", "median"),
        Total_Profit=("Profit", "sum"),
        Avg_Profit_Margin=("Profit", lambda x: (x.sum() / df.loc[x.index, "Revenue"].sum()) * 100),
        Avg_Rating=("Customer_Rating", "mean"),
    ).reset_index().sort_values(by="Total_Revenue", ascending=False)

    cat_summary = cat_summary.round(2)
    cat_summary.to_csv(os.path.join(OUTPUT_DIR, "category_summary_stats.csv"), index=False)
    print(cat_summary.to_string(index=False))

    # Figure 1: Executive Sales & Profitability Dashboard (4-panel)
    df["YearMonth"] = df["Order_Date"].dt.to_period("M").astype(str)
    monthly = df.groupby("YearMonth")[["Revenue", "Profit"]].sum().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Panel A: Monthly Revenue & Profit Trend
    axes[0, 0].plot(monthly["YearMonth"], monthly["Revenue"] / 1000, marker="o", lw=2.5, label="Revenue ($K)", color="#1f77b4")
    axes[0, 0].plot(monthly["YearMonth"], monthly["Profit"] / 1000, marker="s", lw=2.5, label="Profit ($K)", color="#2ca02c")
    axes[0, 0].set_title("A. Monthly Revenue & Profit Trend (2025)", fontsize=13, fontweight="bold")
    axes[0, 0].set_ylabel("Amount ($ Thousands)")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].legend()

    # Panel B: Revenue & Profit by Product Category
    cat_plot = cat_summary.set_index("Category")[["Total_Revenue", "Total_Profit"]] / 1000
    cat_plot.plot(kind="bar", ax=axes[0, 1], color=["#2b5c8f", "#38a3a5"])
    axes[0, 1].set_title("B. Total Revenue & Profit by Product Category ($K)", fontsize=13, fontweight="bold")
    axes[0, 1].set_ylabel("Amount ($ Thousands)")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].legend(["Total Revenue ($K)", "Total Profit ($K)"])

    # Panel C: Regional Revenue Contribution by Customer Segment
    reg_seg = df.pivot_table(index="Region", columns="Customer_Segment", values="Revenue", aggfunc="sum") / 1000
    reg_seg.plot(kind="barh", stacked=True, ax=axes[1, 0], colormap="viridis")
    axes[1, 0].set_title("C. Regional Revenue Breakdown by Customer Segment ($K)", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Revenue ($ Thousands)")
    axes[1, 0].set_ylabel("")

    # Panel D: Impact of Discount Rate on Profit Margin (%)
    df["Profit_Margin_Pct"] = (df["Profit"] / df["Revenue"]) * 100
    disc_impact = df.groupby("Discount_Rate")["Profit_Margin_Pct"].mean().reset_index()
    sns.lineplot(data=disc_impact, x=disc_impact["Discount_Rate"] * 100, y="Profit_Margin_Pct",
                 marker="o", lw=2.5, color="#d62728", ax=axes[1, 1])
    axes[1, 1].axhline(0, color="black", linestyle="--", lw=1)
    axes[1, 1].set_title("D. Discount Rate vs. Average Profit Margin (%)", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Discount Rate (%)")
    axes[1, 1].set_ylabel("Average Profit Margin (%)")

    fig.suptitle("Executive Retail Sales & Profitability Analysis", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "executive_sales_dashboard.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Customer Behavior & Feature Correlation Analysis
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(data=df, x="Customer_Segment", y="Revenue", hue="Sales_Channel", ax=axes2[0], palette="Set2")
    axes2[0].set_title("Order Value Distribution by Segment & Channel", fontsize=13, fontweight="bold")
    axes2[0].set_xlabel("Customer Segment")
    axes2[0].set_ylabel("Order Revenue ($)")

    num_cols = ["Unit_Price", "Quantity", "Discount_Rate", "Marketing_Spend", "Customer_Tenure_Months", "Revenue", "Profit"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes2[1], cbar=False)
    axes2[1].set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")

    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "customer_behavior_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"\nSaved 'executive_sales_dashboard.png' and 'customer_behavior_analysis.png' to ./{OUTPUT_DIR}/")


def run_predictive_modeling(df):
    """Trains and evaluates regression models to predict order Revenue from domain features."""
    print("\n" + "=" * 65)
    print("2. END-TO-END PREDICTIVE MODELING (REVENUE FORECASTING)")
    print("=" * 65)

    features = [
        "Category", "Region", "Customer_Segment", "Sales_Channel",
        "Unit_Price", "Quantity", "Discount_Rate", "Marketing_Spend",
        "Customer_Tenure_Months", "Customer_Rating"
    ]
    target = "Revenue"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_features = ["Category", "Region", "Customer_Segment", "Sales_Channel"]
    num_features = ["Unit_Price", "Quantity", "Discount_Rate", "Marketing_Spend", "Customer_Tenure_Months", "Customer_Rating"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features),
        ]
    )

    models = {
        "Ridge Linear Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
    }

    metrics_list = []
    predictions = {}
    trained_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics_list.append({
            "Model": name,
            "RMSE ($)": round(rmse, 2),
            "MAE ($)": round(mae, 2),
            "R2_Score": round(r2, 4),
        })
        predictions[name] = y_pred
        trained_pipelines[name] = pipe

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_benchmark_metrics.csv"), index=False)
    print(metrics_df.to_string(index=False))

    # Plot Model Predictions & Feature Importances
    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6))

    best_model_name = "Gradient Boosting Regressor"
    y_pred_best = predictions[best_model_name]
    r2_best = metrics_df.loc[metrics_df["Model"] == best_model_name, "R2_Score"].values[0]
    rmse_best = metrics_df.loc[metrics_df["Model"] == best_model_name, "RMSE ($)"].values[0]

    axes3[0].scatter(y_test, y_pred_best, alpha=0.45, color="#1f77b4", edgecolors="k", linewidth=0.3)
    axes3[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Ideal Fit")
    axes3[0].set_title(f"{best_model_name}: Actual vs. Predicted\nR² = {r2_best:.4f} | RMSE = ${rmse_best:.2f}", fontsize=13, fontweight="bold")
    axes3[0].set_xlabel("Actual Order Revenue ($)")
    axes3[0].set_ylabel("Predicted Order Revenue ($)")
    axes3[0].legend()

    # Extract feature importances from Random Forest
    rf_pipe = trained_pipelines["Random Forest Regressor"]
    ohe_names = rf_pipe.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(cat_features)
    all_feature_names = num_features + list(ohe_names)
    importances = rf_pipe.named_steps["model"].feature_importances_

    feat_imp_df = pd.DataFrame({"Feature": all_feature_names, "Importance": importances}).sort_values(by="Importance", ascending=True).tail(10)
    axes3[1].barh(feat_imp_df["Feature"], feat_imp_df["Importance"], color="#2ca02c")
    axes3[1].set_title("Top 10 Predictive Features (Random Forest)", fontsize=13, fontweight="bold")
    axes3[1].set_xlabel("Relative Feature Importance")

    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, "sales_prediction_evaluation.png"), dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"\nSaved 'sales_prediction_evaluation.png' and 'model_benchmark_metrics.csv' to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    sales_df = generate_or_load_dataset()
    perform_exploratory_analysis(sales_df)
    run_predictive_modeling(sales_df)
