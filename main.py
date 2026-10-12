from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = DATA_DIR / "energy.csv"
MODEL_PATH = MODEL_DIR / "energy_forecast_model.pkl"
PREDICTIONS_PATH = OUTPUT_DIR / "forecast_next_24_hours.csv"
METRICS_PATH = OUTPUT_DIR / "evaluation_metrics.txt"

for folder in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="AI Energy Forecasting", page_icon="⚡", layout="wide")
st.title("⚡ AI-Powered Energy Consumption Forecasting")
st.caption("Student project • Simulated hourly energy data • Not a production grid-control system")

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        # Fallback creates a self-contained simulated dataset if the CSV was removed.
        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=24*180, freq="h")
        h, dow, doy = dates.hour.to_numpy(), dates.dayofweek.to_numpy(), dates.dayofyear.to_numpy()
        usage = 45 + 12*np.sin((h-7)*2*np.pi/24) + np.where((dow < 5) & (h >= 8) & (h <= 18), 9, 0)
        usage += 4*np.sin(doy*2*np.pi/365) + rng.normal(0, 2, len(dates))
        pd.DataFrame({"Datetime": dates, "Energy_kWh": np.maximum(5, usage)}).to_csv(DATA_PATH, index=False)
    data = pd.read_csv(DATA_PATH)
    if not {"Datetime", "Energy_kWh"}.issubset(data.columns):
        raise ValueError("energy.csv must contain Datetime and Energy_kWh columns.")
    data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")
    data["Energy_kWh"] = pd.to_numeric(data["Energy_kWh"], errors="coerce")
    data = data.dropna(subset=["Datetime", "Energy_kWh"]).sort_values("Datetime")
    data = data.drop_duplicates(subset=["Datetime"]).set_index("Datetime").resample("h").mean()
    data["Energy_kWh"] = data["Energy_kWh"].interpolate(limit_direction="both").clip(lower=0)
    if len(data) < 100:
        raise ValueError("Please provide at least 100 hourly rows in data/energy.csv.")
    return data

def make_features(data):
    frame = data.copy()
    ix = frame.index
    frame["hour"] = ix.hour
    frame["dayofweek"] = ix.dayofweek
    frame["month"] = ix.month
    frame["dayofyear"] = ix.dayofyear
    frame["is_weekend"] = (ix.dayofweek >= 5).astype(int)
    frame["lag_1"] = frame["Energy_kWh"].shift(1)
    frame["lag_24"] = frame["Energy_kWh"].shift(24)
    frame["rolling_mean_24"] = frame["Energy_kWh"].shift(1).rolling(24).mean()
    return frame.dropna()

FEATURES = ["hour", "dayofweek", "month", "dayofyear", "is_weekend",
            "lag_1", "lag_24", "rolling_mean_24"]

try:
    data = load_data()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

st.sidebar.header("Project controls")
st.sidebar.write("Dataset: `data/energy.csv`")
test_percent = st.sidebar.slider("Test data (%)", min_value=10, max_value=30, value=20, step=5)
forecast_hours = st.sidebar.slider("Forecast horizon (hours)", min_value=6, max_value=48, value=24, step=6)

features_df = make_features(data)
split_at = int(len(features_df) * (1 - test_percent/100))
train, test = features_df.iloc[:split_at], features_df.iloc[split_at:]
if len(train) < 30 or len(test) < 10:
    st.error("Not enough data after feature engineering. Add more hourly rows to energy.csv.")
    st.stop()

@st.cache_resource
def train_model(x_values, y_values):
    model = RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
    model.fit(np.asarray(x_values), np.asarray(y_values))
    return model

# Use tuples to make cache inputs hashable and deterministic.
model = train_model(tuple(map(tuple, train[FEATURES].to_numpy())),
                    tuple(train["Energy_kWh"].to_numpy()))
pred_test = model.predict(test[FEATURES])
mae = mean_absolute_error(test["Energy_kWh"], pred_test)
rmse = float(np.sqrt(mean_squared_error(test["Energy_kWh"], pred_test)))
r2 = r2_score(test["Energy_kWh"], pred_test)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hourly records", f"{len(data):,}")
c2.metric("MAE", f"{mae:.2f} kWh")
c3.metric("RMSE", f"{rmse:.2f} kWh")
c4.metric("R²", f"{r2:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model evaluation", "Forecast", "Dataset"])
with tab1:
    st.subheader("Historical energy consumption")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data.index, data["Energy_kWh"], linewidth=0.8)
    ax.set_ylabel("Energy (kWh)")
    ax.set_xlabel("Datetime")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
    st.info("This repository includes simulated data for learning and demonstration. Replace it with properly licensed real data for a real-world study.")

with tab2:
    st.subheader("Actual vs predicted — chronological holdout")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test.index, test["Energy_kWh"], label="Actual", linewidth=1)
    ax.plot(test.index, pred_test, label="Predicted", linewidth=1)
    ax.set_ylabel("Energy (kWh)")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
    st.write({"MAE_kWh": round(float(mae), 4), "RMSE_kWh": round(rmse, 4), "R2": round(float(r2), 4)})
    if st.button("Save model and evaluation report"):
        joblib.dump(model, MODEL_PATH)
        METRICS_PATH.write_text(
            "AI-Powered Energy Consumption Forecasting\\n"
            f"Rows: {len(data)}\\nTest percentage: {test_percent}\\n"
            f"MAE (kWh): {mae:.4f}\\nRMSE (kWh): {rmse:.4f}\\nR2: {r2:.4f}\\n"
            "Dataset note: simulated hourly data; metrics are educational, not operational validation.\\n",
            encoding="utf-8")
        st.success(f"Saved model to {MODEL_PATH} and report to {METRICS_PATH}")

with tab3:
    st.subheader(f"Next {forecast_hours} hours (recursive forecast)")
    history = data["Energy_kWh"].copy()
    future_dates, future_values = [], []
    for _ in range(forecast_hours):
        next_time = history.index[-1] + pd.Timedelta(hours=1)
        recent = history
        row = {
            "hour": next_time.hour,
            "dayofweek": next_time.dayofweek,
            "month": next_time.month,
            "dayofyear": next_time.dayofyear,
            "is_weekend": int(next_time.dayofweek >= 5),
            "lag_1": float(recent.iloc[-1]),
            "lag_24": float(recent.iloc[-24] if len(recent) >= 24 else recent.iloc[-1]),
            "rolling_mean_24": float(recent.iloc[-24:].mean())
        }
        value = max(0.0, float(model.predict(pd.DataFrame([row])[FEATURES])[0]))
        future_dates.append(next_time)
        future_values.append(value)
        history.loc[next_time] = value
    forecast = pd.DataFrame({"Datetime": future_dates, "Predicted_Energy_kWh": np.round(future_values, 3)})
    fig, ax = plt.subplots(figsize=(12, 4))
    recent_data = data.iloc[-min(72, len(data)):]
    ax.plot(recent_data.index, recent_data["Energy_kWh"], label="Recent actual", linewidth=1.2)
    ax.plot(forecast["Datetime"], forecast["Predicted_Energy_kWh"], label="Forecast", marker="o", markersize=3)
    ax.set_ylabel("Energy (kWh)")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(forecast, use_container_width=True)
    st.download_button("Download forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
                       file_name="forecast_next_hours.csv", mime="text/csv")
    if st.button("Save forecast to outputs folder"):
        forecast.to_csv(PREDICTIONS_PATH, index=False)
        st.success(f"Saved forecast to {PREDICTIONS_PATH}")

with tab4:
    st.subheader("Dataset preview")
    st.write(f"Date range: {data.index.min()} to {data.index.max()}")
    st.dataframe(data.reset_index().head(30), use_container_width=True)
    st.download_button("Download cleaned dataset", data.reset_index().to_csv(index=False).encode("utf-8"),
                       file_name="cleaned_energy_data.csv", mime="text/csv")

st.divider()
st.caption("Educational prototype only. Forecasts depend on the included simulated data and should not be used for billing, safety, grid dispatch, or operational decisions.")
