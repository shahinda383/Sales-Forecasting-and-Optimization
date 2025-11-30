# ============================================================
# 💥 PHASE 11️⃣ : AutoML Optimization (Optuna) - Modular Pipeline
# Project: Sales Forecasting & Optimization
# ============================================================

import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import gc
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# ⚙ GLOBAL SETTINGS
# ============================================================
N_TRIALS = 20
N_JOBS = 1
LSTM_TUNE_EPOCHS = 5
LSTM_FINAL_EPOCHS = 20
tf.keras.backend.clear_session()


# ============================================================
# 1️⃣ Load Data
# ============================================================
def load_data(file_path="Dataset_Final_Clean_For_Modeling.csv", target_col="Sales", test_size=0.2):
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, shuffle=False)
    print(f"✅ Data Loaded: {X.shape}")
    return X_train, X_valid, y_train, y_valid, X, y


# ============================================================
# 2️⃣ Objective Function for Optuna
# ============================================================
def objective(trial, X_train, X_valid, y_train, y_valid):
    gc.collect()
    tf.keras.backend.clear_session()

    model_type = trial.suggest_categorical("model_type", ["xgboost", "lightgbm", "lstm"])

    if model_type == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "tree_method": "hist",
            "random_state": 42
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)

    elif model_type == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 1.0),
            "device_type": "cpu",
            "random_state": 42
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)

    else:
        n_units = trial.suggest_int("n_units", 32, 96)
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.4)
        lr = trial.suggest_float("lr", 0.001, 0.005)

        model = Sequential([
            LSTM(n_units, input_shape=(1, X_train.shape[1]), return_sequences=False),
            Dropout(dropout_rate),
            Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss="mse", optimizer=optimizer)

        X_train_lstm = np.expand_dims(X_train.values, axis=1)
        X_valid_lstm = np.expand_dims(X_valid.values, axis=1)
        model.fit(X_train_lstm, y_train, epochs=LSTM_TUNE_EPOCHS, batch_size=64, verbose=0)
        preds = model.predict(X_valid_lstm).flatten()
        del X_train_lstm, X_valid_lstm

    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    del model, preds
    gc.collect()
    return rmse


# ============================================================
# 3️⃣ Run Optuna Study
# ============================================================
def run_optimization(X_train, X_valid, y_train, y_valid):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X_train, X_valid, y_train, y_valid), n_trials=N_TRIALS, n_jobs=N_JOBS)
    return study


# ============================================================
# 4️⃣ Train Final Model
# ============================================================
def train_final_model(study, X, y):
    best_model_type = study.best_trial.params["model_type"]

    if best_model_type == "xgboost":
        best_params = {k: v for k, v in study.best_trial.params.items() if k != "model_type"}
        best_params.update({"tree_method": "hist"})
        final_model = XGBRegressor(**best_params)
        final_model.fit(X, y)
        joblib.dump(final_model, "BestModel_XGBoost.pkl")

    elif best_model_type == "lightgbm":
        best_params = {k: v for k, v in study.best_trial.params.items() if k != "model_type"}
        best_params.update({"device_type": "cpu"})
        final_model = LGBMRegressor(**best_params)
        final_model.fit(X, y)
        joblib.dump(final_model, "BestModel_LightGBM.pkl")

    else:
        n_units = study.best_trial.params["n_units"]
        dropout_rate = study.best_trial.params["dropout"]
        lr = study.best_trial.params["lr"]

        model = Sequential([
            LSTM(n_units, input_shape=(1, X.shape[1]), return_sequences=False),
            Dropout(dropout_rate),
            Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss="mse", optimizer=optimizer)
        X_lstm = np.expand_dims(X.values, axis=1)
        model.fit(X_lstm, y, epochs=LSTM_FINAL_EPOCHS, batch_size=64, verbose=1)
        model.save("BestModel_LSTM.h5")
        del X_lstm

    print("✅ Saved the Best Model Successfully!")
    return best_model_type


# ============================================================
# 5️⃣ Main Execution
# ============================================================
if __name__ == "__main__":
    X_train, X_valid, y_train, y_valid, X, y = load_data()
    study = run_optimization(X_train, X_valid, y_train, y_valid)

    print("\n🎯 Best Model Type:", study.best_trial.params["model_type"])
    print("🔥 Best RMSE:", study.best_value)
    print("🧩 Best Parameters:")
    for key, value in study.best_trial.params.items():
        print(f"   {key}: {value}")

    # Save optimization results
    optuna_df = study.trials_dataframe()
    optuna_df.to_csv("AutoML_Optimization_Results.csv", index=False)
    print("\n💾 Saved optimization results to AutoML_Optimization_Results.csv")

    # Train final model
    best_model_type = train_final_model(study, X, y)

    # Save summary
    summary = pd.DataFrame({
        "Best_Model": [best_model_type],
        "Best_RMSE": [study.best_value],
        "Best_Params": [study.best_trial.params]
    })
    summary.to_csv("BestModel_Summary.csv", index=False)
    print("\n💾 Summary saved to BestModel_Summary.csv")
    print("\n🚀 AutoML Optimization Completed Successfully ✅")