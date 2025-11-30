# ============================================================
# 🚀 PHASE 3️⃣ : ML Engineering - Day 3
# Project: Sales Forecasting & Optimization
# Task: Time Series Modeling (ARIMA / SARIMA)
# ============================================================

# ==============================
# 📦 1. Import Required Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ==============================
# ⚙ 2. GPU Acceleration Check
# ==============================
def check_gpu():
    try:
        import cudf
        print("🚀 GPU acceleration enabled using cuDF.")
        return True
    except:
        print("⚠ cuDF not found, running on CPU mode.")
        return False


# ==============================
# 📂 3. Load Dataset
# ==============================
def load_dataset(path="Dataset_Final_Clean_For_Modeling.csv"):
    df = pd.read_csv(path)
    print("✅ Dataset loaded successfully!")
    return df


# ==============================
# 🕓 4. Create Time Index
# ==============================
def create_time_index(df, time_level='Week'):
    time_cols = ['Year', 'Month', 'Week', 'Day']
    available = [c for c in time_cols if c in df.columns]

    if not available:
        raise ValueError("❌ No time-based columns found!")

    if time_level == 'Week':
        df['Date'] = pd.to_datetime(df[['Year', 'Week']].astype(str).agg('-'.join, axis=1) + '-1',
                                    format='%Y-%U-%w')
    elif time_level == 'Month':
        df['Date'] = pd.to_datetime(df[['Year', 'Month']].astype(str).agg('-'.join, axis=1) + '-1',
                                    format='%Y-%m-%d')
    elif time_level == 'Day':
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    else:
        raise ValueError("⚠ Invalid time level. Choose: 'Day','Week','Month'")

    df = df.sort_values('Date')
    print(f"✅ Time index created using {time_level}.")
    print(f"📅 Range: {df['Date'].min()} → {df['Date'].max()}")
    return df


# ==============================
# 🏬 5. Extract Time Series for a Store
# ==============================
def extract_store_series(df):
    store_id = df['Store'].unique()[0]
    series = df[df['Store'] == store_id][['Date', 'Sales']].set_index('Date')
    print(f"✅ Time-series prepared for Store {store_id}")
    return series, store_id


# ==============================
# 📊 6. Visualization
# ==============================
def visualize_series(df_store, store_id):
    plt.figure(figsize=(12,5))
    sns.lineplot(x=df_store.index, y=df_store['Sales'])
    plt.title(f"🕓 Sales Over Time - Store {store_id}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.show()


# ==============================
# 🔍 7. Stationarity Check
# ==============================
def check_stationarity(series):
    result = adfuller(series)
    print("📉 ADF Test:")
    print(f"🔹 ADF Statistic: {result[0]:.3f}")
    print(f"🔹 p-value: {result[1]:.3f}")

    if result[1] < 0.05:
        print("✅ Series is stationary.")
        return False
    else:
        print("⚠ Not stationary → Differencing required.")
        return True


# ==============================
# 🧩 8. Differencing
# ==============================
def apply_differencing(df_store, need_diff):
    if need_diff:
        df_store['Sales_diff'] = df_store['Sales'].diff().dropna()
    else:
        df_store['Sales_diff'] = df_store['Sales']

    plt.figure(figsize=(10,5))
    plt.plot(df_store['Sales_diff'], color="purple")
    plt.title("Differenced Series")
    plt.show()

    return df_store


# ==============================
# 📈 9. ACF & PACF
# ==============================
def plot_acf_pacf(df_store):
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    plot_acf(df_store['Sales_diff'].dropna(), ax=axes[0], lags=30)
    plot_pacf(df_store['Sales_diff'].dropna(), ax=axes[1], lags=30)
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    plt.show()


# ==============================
# 🧠 10. Train ARIMA
# ==============================
def train_arima(df_store):
    print("🚀 Training ARIMA(2,1,2)...")
    model = sm.tsa.ARIMA(df_store['Sales'], order=(2,1,2))
    result = model.fit()
    print(result.summary())
    return result


# ==============================
# 🌍 11. Train SARIMA
# ==============================
def train_sarima(df_store):
    print("🚀 Training SARIMA...")
    model = sm.tsa.statespace.SARIMAX(
        df_store['Sales'],
        order=(2,1,2),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)
    print("✅ SARIMA trained!")
    return result


# ==============================
# 🔮 12. Forecast 30 Steps
# ==============================
def forecast_next(arima_model, sarima_model):
    forecast_arima = arima_model.forecast(steps=30)
    forecast_sarima = sarima_model.forecast(steps=30)
    return forecast_arima, forecast_sarima


# ==============================
# 📊 13. Plot Forecasts
# ==============================
def plot_forecasts(df_store, forecast_arima, forecast_sarima, store_id):
    plt.figure(figsize=(12,6))
    plt.plot(df_store['Sales'], label="Actual", color="black", linewidth=2)
    plt.plot(forecast_arima.index, forecast_arima, "--", label="ARIMA", color="blue")
    plt.plot(forecast_sarima.index, forecast_sarima, "--", label="SARIMA", color="red")
    plt.title(f"📈 Forecast - Store {store_id}")
    plt.legend()
    plt.show()


# ==============================
# 🧾 14. Evaluate Models
# ==============================
def evaluate(df_store, forecast_arima, forecast_sarima):
    test_data = df_store['Sales'][-30:]

    arima_eval = {
        "Model": "ARIMA",
        "MAE": mean_absolute_error(test_data, forecast_arima),
        "RMSE": np.sqrt(mean_squared_error(test_data, forecast_arima)),
        "R²": r2_score(test_data, forecast_arima)
    }

    sarima_eval = {
        "Model": "SARIMA",
        "MAE": mean_absolute_error(test_data, forecast_sarima),
        "RMSE": np.sqrt(mean_squared_error(test_data, forecast_sarima)),
        "R²": r2_score(test_data, forecast_sarima)
    }

    results = pd.DataFrame([arima_eval, sarima_eval]).set_index("Model")
    print("📊 Model Performance:")
    print(results.round(4))
    return results


# ==============================
# 🧠 15. Residual Analysis
# ==============================
def residual_analysis(sarima_result):
    residuals = sarima_result.resid

    plt.figure(figsize=(10,4))
    sns.histplot(residuals, bins=40, kde=True, color="orange")
    plt.title("Residual Distribution (SARIMA)")
    plt.show()

    sm.graphics.tsa.plot_acf(residuals, lags=30)
    plt.title("Residual ACF")
    plt.show()


# ==============================
# 💾 16. Save Model
# ==============================
def save_model(result, path="sarima_sales_model.pkl"):
    result.save(path)
    print("💾 SARIMA model saved!")


# ==============================
# 🏁 17. MAIN PIPELINE
# ==============================
def run_pipeline():
    check_gpu()

    df = load_dataset()
    df = create_time_index(df, time_level="Week")
    df_store, store_id = extract_store_series(df)

    visualize_series(df_store, store_id)

    need_diff = check_stationarity(df_store['Sales'])
    df_store = apply_differencing(df_store, need_diff)

    plot_acf_pacf(df_store)

    arima_model = train_arima(df_store)
    sarima_model = train_sarima(df_store)

    forecast_arima, forecast_sarima = forecast_next(arima_model, sarima_model)

    plot_forecasts(df_store, forecast_arima, forecast_sarima, store_id)

    results = evaluate(df_store, forecast_arima, forecast_sarima)

    residual_analysis(sarima_model)

    save_model(sarima_model)

    print("🏆 Best Model:", results["R²"].idxmax())
    print("🎯 Day 3 Completed Successfully!")


if __name__ == "__main__":
    run_pipeline()