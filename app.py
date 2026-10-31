import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Real-World Retail Data Analytics & Prediction", layout="wide")

st.title("📊 Real-World Retail Sales Analytics & Revenue Prediction")
st.markdown(
    "Explore end-to-end **domain-specific retail performance**, filter by region and product category, "
    "and run **live revenue forecasting** using supervised machine learning."
)

@st.cache_data
def load_data():
    filepath = os.path.join("data", "retail_sales_data.csv")
    return pd.read_csv(filepath, parse_dates=["Order_Date"])

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Dataset")
selected_regions = st.sidebar.multiselect("Select Regions", options=df["Region"].unique(), default=list(df["Region"].unique()))
selected_categories = st.sidebar.multiselect("Select Categories", options=df["Category"].unique(), default=list(df["Category"].unique()))

filtered_df = df[df["Region"].isin(selected_regions) & df["Category"].isin(selected_categories)]

# KPI Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total_rev = filtered_df["Revenue"].sum()
total_prof = filtered_df["Profit"].sum()
avg_margin = (total_prof / total_rev * 100) if total_rev > 0 else 0.0
total_orders = len(filtered_df)

kpi1.metric("Total Revenue", f"${total_rev:,.2f}")
kpi2.metric("Total Profit", f"${total_prof:,.2f}")
kpi3.metric("Net Profit Margin", f"{avg_margin:.2f}%")
kpi4.metric("Total Orders", f"{total_orders:,}")

st.markdown("---")

# Visual Row
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Revenue & Profit by Category")
    cat_agg = filtered_df.groupby("Category")[["Revenue", "Profit"]].sum() / 1000
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    cat_agg.plot(kind="bar", ax=ax1, color=["#2b5c8f", "#38a3a5"])
    ax1.set_ylabel("Amount ($ Thousands)")
    ax1.tick_params(axis="x", rotation=25)
    st.pyplot(fig1)

with col2:
    st.subheader("2. Discount Rate Impact on Profit Margin")
    filtered_df = filtered_df.copy()
    filtered_df["Margin_Pct"] = (filtered_df["Profit"] / filtered_df["Revenue"]) * 100
    disc_df = filtered_df.groupby("Discount_Rate")["Margin_Pct"].mean().reset_index()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=disc_df, x=disc_df["Discount_Rate"] * 100, y="Margin_Pct", marker="o", lw=2.5, color="#d62728", ax=ax2)
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Discount Rate (%)")
    ax2.set_ylabel("Avg Profit Margin (%)")
    st.pyplot(fig2)

st.markdown("---")
st.subheader("3. Interactive Order Revenue Predictor")

p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    in_cat = st.selectbox("Category", df["Category"].unique())
    in_reg = st.selectbox("Region", df["Region"].unique())
    in_seg = st.selectbox("Customer Segment", df["Customer_Segment"].unique())
with p_col2:
    in_chan = st.selectbox("Sales Channel", df["Sales_Channel"].unique())
    in_price = st.number_input("Unit Price ($)", min_value=10.0, max_value=1200.0, value=120.0, step=5.0)
    in_qty = st.slider("Quantity", 1, 10, 2)
with p_col3:
    in_disc = st.selectbox("Discount Rate", [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], index=1)
    in_mkt = st.slider("Marketing Spend ($)", 2.0, 50.0, 15.0)
    in_tenure = st.slider("Customer Tenure (Months)", 1, 60, 18)

features = [
    "Category", "Region", "Customer_Segment", "Sales_Channel",
    "Unit_Price", "Quantity", "Discount_Rate", "Marketing_Spend",
    "Customer_Tenure_Months", "Customer_Rating"
]
cat_features = ["Category", "Region", "Customer_Segment", "Sales_Channel"]
num_features = ["Unit_Price", "Quantity", "Discount_Rate", "Marketing_Spend", "Customer_Tenure_Months", "Customer_Rating"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features),
])

model_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42)),
])
model_pipe.fit(df[features], df["Revenue"])

input_df = pd.DataFrame([{
    "Category": in_cat,
    "Region": in_reg,
    "Customer_Segment": in_seg,
    "Sales_Channel": in_chan,
    "Unit_Price": in_price,
    "Quantity": in_qty,
    "Discount_Rate": in_disc,
    "Marketing_Spend": in_mkt,
    "Customer_Tenure_Months": in_tenure,
    "Customer_Rating": 4.4,
}])

pred_rev = model_pipe.predict(input_df)[0]
st.success(f"**Predicted Net Order Revenue:** `${pred_rev:,.2f}`")
