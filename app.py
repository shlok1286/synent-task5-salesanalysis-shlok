import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import Ridge

DATA_FILE = "Sample - Superstore.csv"
MODEL_FILE = "sales_prediction_model.pkl"


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    df = df.drop_duplicates().copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date"]).copy()
    df["Month"] = df["Order Date"].dt.month
    df["Month_Year"] = df["Order Date"].dt.to_period("M").astype(str)
    df["Profit Margin"] = (df["Profit"] / df["Sales"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = ["Ship Mode", "Region", "Category", "Sub-Category", "Quantity", "Discount", "Month"]
    return df[features + ["Sales"]].copy()


def train_sales_model(df: pd.DataFrame) -> dict:
    model_frame = build_model_frame(df)
    X = pd.get_dummies(model_frame.drop(columns=["Sales"]), drop_first=True)
    y = model_frame["Sales"]
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    model_data = {"model": model, "columns": X.columns.tolist()}
    joblib.dump(model_data, MODEL_FILE)
    return model_data


@st.cache_resource(show_spinner=False)
def load_model(df: pd.DataFrame) -> dict:
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return train_sales_model(df)


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def create_prediction_input(selection: dict, feature_columns: list) -> pd.DataFrame:
    input_df = pd.DataFrame([selection])
    input_df = pd.get_dummies(input_df)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[feature_columns]


def render_metrics(df: pd.DataFrame) -> None:
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    avg_discount = df["Discount"].mean()
    avg_order_value = df["Sales"].mean()
    profitable_category = df.groupby("Category")["Profit"].sum().idxmax()
    best_region = df.groupby("Region")["Sales"].sum().idxmax()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", format_currency(total_sales))
    col2.metric("Total Profit", format_currency(total_profit))
    col3.metric("Average Order Value", format_currency(avg_order_value))
    col4.metric("Average Discount", f"{avg_discount:.2%}")

    col5, col6 = st.columns(2)
    col5.info(f"Top Category by Profit: **{profitable_category}**")
    col6.info(f"Top Region by Sales: **{best_region}**")


def render_visualizations(df: pd.DataFrame) -> None:
    st.subheader("Interactive Sales Visualizations")

    region_sales = df.groupby("Region")["Sales"].sum().reset_index().sort_values(by="Sales", ascending=False)
    fig_region = px.bar(region_sales, x="Region", y="Sales", color="Region", title="Sales by Region", text_auto='.2s')
    st.plotly_chart(fig_region, use_container_width=True)

    category_profit = df.groupby("Category")["Profit"].sum().reset_index().sort_values(by="Profit", ascending=False)
    fig_profit = px.bar(category_profit, x="Category", y="Profit", color="Category", title="Profit by Category", text_auto='.2s')
    st.plotly_chart(fig_profit, use_container_width=True)

    monthly_sales = df.groupby("Month_Year")["Sales"].sum().reset_index()
    monthly_sales = monthly_sales.sort_values("Month_Year")
    fig_monthly = px.line(monthly_sales, x="Month_Year", y="Sales", markers=True, title="Monthly Sales Trend")
    fig_monthly.update_layout(xaxis_title="Month-Year", yaxis_title="Sales", xaxis_tickangle=-45)
    st.plotly_chart(fig_monthly, use_container_width=True)

    top_subcategory = df.groupby("Sub-Category")["Sales"].sum().reset_index().sort_values(by="Sales", ascending=False).head(10)
    fig_top = px.bar(top_subcategory, x="Sub-Category", y="Sales", title="Top 10 Sub-Category Sales", text_auto='.2s')
    st.plotly_chart(fig_top, use_container_width=True)

    corr_df = df.select_dtypes(include=["int64", "float64"]).corr()
    fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)


def render_data_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset Summary")
    st.write("This dataset is the same Sample Superstore data used for the internship analysis.")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("**Columns included:**")
    st.write(df.columns.tolist())
    st.write("---")
    st.subheader("Sales and Profit Summary")
    st.write(df.describe(include="all").T)


def render_prediction_panel(df: pd.DataFrame, model_data: dict) -> None:
    st.subheader("Sales Forecasting Simulator")
    st.write(
        "Estimate expected sales for a new order using the dataset features and a lightweight regression model. "
        "The model is trained from the same Superstore dataset on first use and saved locally."
    )

    ship_mode = st.selectbox("Ship Mode", sorted(df["Ship Mode"].unique()))
    region = st.selectbox("Region", sorted(df["Region"].unique()))
    category = st.selectbox("Category", sorted(df["Category"].unique()))
    sub_category = st.selectbox("Sub-Category", sorted(df["Sub-Category"].unique()))
    quantity = st.number_input("Quantity", min_value=1, max_value=1000, value=5)
    discount = st.slider("Discount", min_value=0.0, max_value=0.8, value=0.1, step=0.05)
    order_date = st.date_input("Order Date")
    month = order_date.month

    if st.button("Predict Sales"):
        form_data = {
            "Ship Mode": ship_mode,
            "Region": region,
            "Category": category,
            "Sub-Category": sub_category,
            "Quantity": quantity,
            "Discount": discount,
            "Month": month,
        }
        x_input = create_prediction_input(form_data, model_data["columns"])
        prediction = model_data["model"].predict(x_input)[0]
        st.success(f"Predicted Sales: {format_currency(prediction)}")
        st.info("Use this forecast to compare customer segments, product categories, and shipping choices.")


def main() -> None:
    st.set_page_config(
        page_title="Sales Analysis Dashboard",
        page_icon="📈",
        layout="wide",
    )

    st.title("Sales Data Analysis Dashboard")
    st.markdown(
        "Explore the sample superstore dataset with interactive charts, business metrics, and a sales forecasting simulator. "
        "This dashboard preserves the original internship analysis while adding a reusable Streamlit presentation layer."
    )

    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset not found: {DATA_FILE}. Please ensure the CSV file is present in the repository root.")
        return

    df = load_data(DATA_FILE)
    model_data = load_model(df)

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Choose a view", ["Executive Summary", "Visualizations", "Forecasting"])
    st.sidebar.markdown("---")
    st.sidebar.write("**Data source:** Sample Superstore CSV")
    st.sidebar.write("**Model file:** sales_prediction_model.pkl")

    if page == "Executive Summary":
        st.header("Executive Summary")
        render_metrics(df)
        render_data_overview(df)

    elif page == "Visualizations":
        st.header("Visual Data Insights")
        render_visualizations(df)

    else:
        st.header("Sales Forecasting")
        render_prediction_panel(df, model_data)

    st.markdown("---")
    st.caption("Created to preserve the existing internship work while adding a production-ready Streamlit interface.")


if __name__ == "__main__":
    main()
