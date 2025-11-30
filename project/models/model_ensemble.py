# ===============================================================
# 🌟 PHASE 9 : Ensemble Model (Weighted Average)
# Project: Sales Forecasting & Optimization
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import os, warnings
warnings.filterwarnings('ignore')


# ==============================================
# 🔹 Functions
# ==============================================
def prepare_for_merge(df_pred, pred_col):
    """Aggregate predictions by Date"""
    return df_pred.groupby('Date', as_index=False)[pred_col].mean()


def load_lstm_predictions(df, model_path="lstm_model.h5", seq_len=12):
    """Load LSTM model, generate predictions, and return dataframe with Date & prediction"""
    custom_objects = {
        'mse': tf.keras.metrics.MeanSquaredError(),
        'MeanSquaredError': tf.keras.metrics.MeanSquaredError,
    }

    lstm_model = load_model(model_path, custom_objects=custom_objects)
    
    sales = df['Sales'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(sales)

    X_lstm = np.array([sales_scaled[i:i+seq_len] for i in range(len(sales_scaled)-seq_len)])
    preds = lstm_model.predict(X_lstm, verbose=0)
    preds = scaler.inverse_transform(preds)

    dates = df['Date'].iloc[seq_len:].reset_index(drop=True)
    df_pred = pd.DataFrame({'Date': dates, 'LSTM_Prediction': preds.flatten()})
    df_pred.to_csv("predictions_LSTM.csv", index=False)
    print("✅ LSTM predictions saved as predictions_LSTM.csv")
    return df_pred


def generate_prophet_predictions(df):
    """Generate Prophet predictions if CSV does not exist"""
    try:
        df_prophet = pd.read_csv("predictions_Prophet.csv")
        print("✅ Prophet predictions loaded successfully.")
    except FileNotFoundError:
        print("⚠ Prophet CSV not found — generating predictions...")
        prophet_df = df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
        model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model_prophet.fit(prophet_df)
        future = prophet_df[['ds']]
        forecast = model_prophet.predict(future)
        df_prophet = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Prophet_Prediction'})
        df_prophet.to_csv("predictions_Prophet.csv", index=False)
        print("✅ Prophet predictions saved as predictions_Prophet.csv")
    return df_prophet


def weighted_ensemble(df_list, weights):
    """Compute weighted average ensemble"""
    df_ensemble = df_list[0]
    for df in df_list[1:]:
        df_ensemble = df_ensemble.merge(df, on='Date', how='inner')
    df_ensemble['Ensemble_Prediction'] = sum(df_ensemble[col] * w for col, w in weights.items())
    return df_ensemble


def evaluate_ensemble(df, target_col='Sales', pred_col='Ensemble_Prediction'):
    """Compute RMSE, MAE, R2"""
    rmse = np.sqrt(mean_squared_error(df[target_col], df[pred_col]))
    mae = mean_absolute_error(df[target_col], df[pred_col])
    r2 = r2_score(df[target_col], df[pred_col])
    print(f"\n📊 Ensemble Performance Results:\nRMSE: {rmse:.4f}\nMAE : {mae:.4f}\nR²  : {r2:.4f}")
    return rmse, mae, r2


# ==============================================
# 🔹 Main Execution
# ==============================================
if __name__ == "__main__":

    print("📂 Loading datasets and predictions...")
    df = pd.read_csv("Dataset_Final_Clean_For_Modeling.csv")
    df = df[['Store', 'Dept', 'Sales', 'IsHoliday', 'Size', 'Temperature_x', 'Fuel_Price_x',
             'MarkDown1', 'MarkDown3', 'MarkDown4', 'CPI', 'Unemployment',
             'Year', 'Month', 'Week']]

    df['Date'] = pd.to_datetime(df['Year'].astype(str) + df['Week'].astype(str) + '1', format='%Y%W%w')
    df = df.sort_values('Date')

    # Load other model predictions
    pred_xgb = pd.read_csv("predictions_XGBoost.csv")
    pred_lgb = pd.read_csv("predictions_LightGBM.csv")

    if 'Date' not in pred_xgb.columns:
        pred_xgb['Date'] = df['Date'].iloc[-len(pred_xgb):].reset_index(drop=True)
    if 'Date' not in pred_lgb.columns:
        pred_lgb['Date'] = df['Date'].iloc[-len(pred_lgb):].reset_index(drop=True)

    # Prophet predictions
    prophet_preds = generate_prophet_predictions(df)

    # LSTM predictions
    pred_lstm = load_lstm_predictions(df)

    # Prepare & align all predictions
    pred_lstm = prepare_for_merge(pred_lstm, 'LSTM_Prediction')
    prophet_preds = prepare_for_merge(prophet_preds, 'Prophet_Prediction')
    pred_xgb = prepare_for_merge(pred_xgb, 'XGBoost_Predicted_Sales')
    pred_lgb = prepare_for_merge(pred_lgb, 'LightGBM_Predicted_Sales')

    for temp_df in [pred_lstm, prophet_preds, pred_xgb, pred_lgb, df]:
        temp_df['Date'] = pd.to_datetime(temp_df['Date'], errors='coerce')

    # Weighted Ensemble
    weights = {
        'XGBoost_Predicted_Sales': 0.35,
        'LightGBM_Predicted_Sales': 0.25,
        'LSTM_Prediction': 0.25,
        'Prophet_Prediction': 0.15
    }

    ensemble_df = weighted_ensemble([pred_lstm, prophet_preds, pred_xgb, pred_lgb], weights)

    # Merge with actual sales
    sales_df = df[['Date', 'Sales']].groupby('Date', as_index=False)['Sales'].mean()
    ensemble_df = ensemble_df.merge(sales_df, on='Date', how='inner')

    # Evaluate
    evaluate_ensemble(ensemble_df)

    # Save results
    os.makedirs("Models", exist_ok=True)
    ensemble_df.to_csv("Models/Ensemble_Model_Results.csv", index=False)
    print("✅ Ensemble results saved successfully in Models/Ensemble_Model_Results.csv")

    # Visualization
    plt.figure(figsize=(14,6))
    plt.plot(ensemble_df['Date'], ensemble_df['Sales'], label='Actual Sales', alpha=0.7)
    plt.plot(ensemble_df['Date'], ensemble_df['Ensemble_Prediction'], label='Ensemble Prediction', alpha=0.9)
    plt.title("📈 Ensemble Model Forecasting (XGBoost + Prophet + LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n✅ Ensemble modeling completed successfully!")