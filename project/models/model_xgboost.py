# ============================================================
# 🌟 XGBoost Pipeline
# Project: Sales Forecasting & Optimization
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import torch
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 💾 Save Results Utility
# ============================================================
def save_results_to_csv(model_name, y_true, y_pred):
    results_df = pd.DataFrame({
        'Actual_Sales': y_true,
        f'{model_name}_Predicted_Sales': y_pred
    })

    filename = f"predictions_{model_name}.csv"
    results_df.to_csv(filename, index=False)
    print(f"💾 Saved: {filename}")


# ============================================================
# 🚀 Main XGBoost Training Function
# ============================================================
def run_xgboost_pipeline():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Running XGBoost on: {device.upper()}")

    # Load dataset
    df = pd.read_csv("Dataset_Final_Clean_For_Modeling.csv")

    # Create date column if needed
    if {'Year', 'Month', 'Week'}.issubset(df.columns):
        df['Date'] = pd.to_datetime(
            df['Year'].astype(str) + '-' +
            df['Month'].astype(str) + '-' +
            df['Week'].astype(str) + '0',
            errors='coerce'
        )
    else:
        raise ValueError("⚠ Missing required time columns.")

    # Select features and target
    target = 'Sales'
    features = [c for c in df.columns if c not in ['Sales', 'Date']]

    X = df[features]
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Move tensors to GPU if available
    if device == "cuda":
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)

    # Train model
    print("\n🚀 Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        tree_method='gpu_hist' if device == "cuda" else 'hist',
        predictor='gpu_predictor' if device == "cuda" else 'cpu_predictor',
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    # Predict
    y_pred = xgb_model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📊 XGBoost Performance:")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    # Save predictions
    save_results_to_csv("XGBoost", y_test, y_pred)

    return xgb_model


# ============================================================
# 🔥 Run
# ============================================================
if __name__ == "__main__":
    run_xgboost_pipeline()