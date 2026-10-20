import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market EDA & Prediction")
st.write("Exploratory Data Analysis using Random Forest Regression")

# Automatically Find CSV
BASE_DIR = Path(__file__).parent
csv_files = list(BASE_DIR.glob("*.csv"))

if not csv_files:
    st.error("CSV file not found. Upload your stock dataset to GitHub.")
    st.stop()

# Select Dataset if Multiple CSV Files Exist
selected_file = st.sidebar.selectbox(
    "Select Stock Dataset",
    csv_files,
    format_func=lambda x: x.name
)

# Load Dataset
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

df = load_data(str(selected_file))

st.success("Dataset Loaded Successfully!")

# Dataset Overview
st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Rows", f"{df.shape[0]:,}")
col2.metric("Total Columns", df.shape[1])
col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")

st.subheader("First 10 Rows")
st.dataframe(df.head(10), use_container_width=True)

with st.expander("Dataset Information"):
    st.write("Column Names:", df.columns.tolist())
    st.write("Missing Values:")
    st.dataframe(df.isnull().sum().to_frame("Missing Values"))

    st.write("Statistical Summary:")
    st.dataframe(df.describe(include="all").T)

# Detect Date Column
date_col = next(
    (
        c for c in df.columns
        if c.lower() in ["date", "datetime", "timestamp"]
    ),
    None
)

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)

# Numeric Data
numeric_df = df.select_dtypes(include=np.number).copy()
numeric_df = numeric_df.dropna(axis=1, how="all")

if numeric_df.empty:
    st.error("No numeric columns found in the dataset.")
    st.stop()

# EDA
st.header("2. Exploratory Data Analysis")

# Stock Price Graph
price_columns = [
    c for c in numeric_df.columns
    if c.lower() in ["open", "high", "low", "close", "price",
                     "stock_price", "closing_price", "close_price"]
]

if price_columns:
    selected_price = st.selectbox(
        "Select Price Column",
        price_columns
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    x_values = (
        df.loc[numeric_df.index, date_col]
        if date_col
        else numeric_df.index
    )

    ax.plot(x_values, numeric_df[selected_price])
    ax.set_title(f"{selected_price} Analysis")
    ax.set_xlabel("Date / Index")
    ax.set_ylabel("Price")
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")

if len(numeric_df.columns) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Not enough numeric columns for correlation analysis.")

# Volume Distribution
volume_col = next(
    (c for c in numeric_df.columns if c.lower() == "volume"),
    None
)

if volume_col:
    st.subheader("Trading Volume Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(numeric_df[volume_col].dropna(), kde=True, ax=ax)

    st.pyplot(fig)
    plt.close(fig)

# Random Forest Model
st.header("3. Random Forest Regression")

default_target = next(
    (
        c for c in numeric_df.columns
        if c.lower() in [
            "close", "closing_price", "close_price",
            "stock_price", "price"
        ]
    ),
    numeric_df.columns[0]
)

target_col = st.selectbox(
    "Select Target Column",
    numeric_df.columns.tolist(),
    index=numeric_df.columns.tolist().index(default_target)
)

# Prepare Features
data = numeric_df.copy()
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=[target_col])

excluded = [target_col]

for col in data.columns:
    if col.lower() in ["adj close", "adj_close"]:
        excluded.append(col)

X = data.drop(columns=excluded)
y = data[target_col]

if X.empty:
    st.error("No numeric input features available for training.")
    st.stop()

# Fill Missing Values
X = X.fillna(X.median())
valid_rows = y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows]

if len(X) < 5:
    st.error("Dataset needs at least 5 valid rows.")
    st.stop()

# Chronological Split
split_index = int(len(X) * 0.8)

if split_index == 0 or split_index >= len(X):
    st.error("Unable to split dataset into training and testing data.")
    st.stop()

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train Model
with st.spinner("Training Random Forest Model..."):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

st.success("Random Forest Model Trained Successfully!")

# Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("MAE", f"{mae:.4f}")
col2.metric("MSE", f"{mse:.4f}")
col3.metric("RMSE", f"{rmse:.4f}")
col4.metric("R² Score", f"{r2:.4f}")

# Actual vs Predicted
st.header("4. Actual vs Predicted Prices")

results = pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": predictions
})

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(results["Actual Price"].values, label="Actual Price")
ax.plot(results["Predicted Price"].values, label="Predicted Price")

ax.set_xlabel("Test Samples")
ax.set_ylabel(target_col)
ax.set_title("Actual vs Predicted Stock Prices")
ax.legend()

st.pyplot(fig)
plt.close(fig)

st.subheader("Prediction Results")
st.dataframe(results.head(50), use_container_width=True)

# Feature Importance
st.header("5. Feature Importance")

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))

importance.plot(kind="bar", ax=ax)

ax.set_title("Random Forest Feature Importance")
ax.set_ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    importance.to_frame(name="Importance"),
    use_container_width=True
)

# Download Results
st.header("6. Download Predictions")

csv_data = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Predictions CSV",
    data=csv_data,
    file_name="stock_predictions.csv",
    mime="text/csv"
)

st.caption(
    "Note: These predictions estimate the selected target from the "
    "available features. They are not guaranteed future stock prices."
)
