# ============================================================
# 🌟 LightGBM Pipeline
# Project: Sales Forecasting & Optimization
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import torch
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 💾 Save Results
# ============================================================
def save_results_to_csv(model_name, y_true, y_pred):
    df_res = pd.DataFrame({
        'Actual_Sales': y_true,
        f'{model_name}_Predicted_Sales': y_pred
    })
    filename = f"predictions_{model_name}.csv"
    df_res.to_csv(filename, index=False)
    print(f"💾 Saved: {filename}")


# ============================================================
# 🚀 LightGBM Pipeline Main Function
# ============================================================
def run_lightgbm_pipeline():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Running LightGBM on: {device.upper()}")

    # Load dataset
    df = pd.read_csv("Dataset_Final_Clean_For_Modeling.csv")

    # Build date column
    if {'Year', 'Month', 'Week'}.issubset(df.columns):
        df['Date'] = pd.to_datetime(
            df['Year'].astype(str) + '-' +
            df['Month'].astype(str) + '-' +
            df['Week'].astype(str) + '0',
            errors='coerce'
        )
    else:
        raise ValueError("⚠ Missing required time columns.")

    # Select features/target
    target = 'Sales'
    features = [c for c in df.columns if c not in ['Sales', 'Date']]

    X = df[features]
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Convert to numpy if GPU
    if device == "cuda":
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)

    # Prepare datasets
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    # Parameters
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'device': 'gpu' if device == "cuda" else 'cpu',
        'num_leaves': 40,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    callbacks_list = [
        lgb.log_evaluation(period=0),
        lgb.early_stopping(stopping_rounds=50, verbose=False)
    ]

    # Train model
    print("\n🚀 Training LightGBM...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=400,
        valid_sets=[train_set, val_set],
        valid_names=['train', 'eval'],
        callbacks=callbacks_list
    )

    # Predict
    preds = model.predict(X_test, num_iteration=model.best_iteration)

    # Evaluate
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n📊 LightGBM Performance:")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    # Save CSV
    save_results_to_csv("LightGBM", y_test, preds)

    return model


# ============================================================
# 🔥 Run
# ============================================================
if __name__ == "__main__":
    run_lightgbm_pipeline()