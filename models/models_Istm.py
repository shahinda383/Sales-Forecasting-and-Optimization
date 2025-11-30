# ==============================================
# 📅 DAY 6: Deep Learning Models (LSTM & GRU)
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import joblib, datetime, warnings
warnings.filterwarnings('ignore')


# ==============================================
# 🔹 Functions
# ==============================================
def build_lstm(seq_len, n_feats):
    model = Sequential([
        LSTM(128, input_shape=(seq_len, n_feats)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_gru(seq_len, n_feats):
    model = Sequential([
        GRU(128, input_shape=(seq_len, n_feats)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def evaluate_model(model, X_test, y_test, y_scaler, name, timestamp):
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_true = y_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100

    print(f"\n📊 {name} Results:")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAPE: {mape:.2f}%")

    pd.DataFrame({'Actual': y_true, 'Predicted': y_pred}).to_csv(f"{name.lower()}_preds_{timestamp}.csv", index=False)
    print(f"💾 Saved {name} predictions to CSV")

    return mae, rmse, mape, y_true, y_pred


# ==============================================
# 🔹 Main Execution
# ==============================================
if __name__ == "__main__":

    print("🚀 Starting Day 6: Deep Learning (LSTM & GRU)\n")

    # Step 1: Load Data
    dataset_path = "Dataset_Final_Clean_For_Modeling.csv"
    data = pd.read_csv(dataset_path)

    # Step 2: Cleaning
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Sales']).dropna()
    lags = ['Sales_Lag_30', 'Sales_Lag_14', 'Sales_Lag_7', 'Sales_Lag_1']
    missing = [c for c in lags + ['Sales'] if c not in data.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    data_seq = data.dropna(subset=lags + ['Sales']).copy()
    print(f"✅ Rows available for training: {len(data_seq):,}\n")

    # Step 3: Prepare sequences
    X_seq = data_seq[lags].values.astype(float)
    y = data_seq['Sales'].values.astype(float)
    X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)
    print(f"📊 X_seq shape: {X_seq.shape}\n")

    # Step 4: Scaling
    X_scaler = MinMaxScaler()
    N, T, F = X_seq.shape
    X_scaled = X_scaler.fit_transform(X_seq.reshape(N,-1)).reshape(N, T, F)

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1,1)).flatten()

    # Step 5: Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42, shuffle=True)
    print(f"📚 Train: {X_train.shape} | Test: {X_test.shape}\n")

    # Step 6: GPU check
    if tf.config.list_physical_devices('GPU'):
        print("⚡ GPU Detected — Training on GPU.\n")
    else:
        print("⚙ Training on CPU.\n")

    # Step 7 & 8: Build & Train
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    seq_len = T

    print("🔹 Training LSTM…")
    lstm_model = build_lstm(seq_len, F)
    history_lstm = lstm_model.fit(X_train, y_train, validation_split=0.15, epochs=40, batch_size=256, callbacks=[es], verbose=2)

    print("\n🔸 Training GRU…")
    gru_model = build_gru(seq_len, F)
    history_gru = gru_model.fit(X_train, y_train, validation_split=0.15, epochs=40, batch_size=256, callbacks=[es], verbose=2)

    # Step 9: Save models
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lstm_model.save(f"lstm_model_{timestamp}.h5")
    gru_model.save(f"gru_model_{timestamp}.h5")
    joblib.dump(X_scaler, f"X_scaler_{timestamp}.pkl")
    joblib.dump(y_scaler, f"y_scaler_{timestamp}.pkl")
    print(f"\n💾 Saved models & scalers — Timestamp {timestamp}\n")

    # Step 10: Evaluate
    mae_lstm, rmse_lstm, mape_lstm, y_true, pred_lstm = evaluate_model(lstm_model, X_test, y_test, y_scaler, "LSTM", timestamp)
    mae_gru, rmse_gru, mape_gru, _, pred_gru = evaluate_model(gru_model, X_test, y_test, y_scaler, "GRU", timestamp)

    # Step 11: Compare
    results = pd.DataFrame({
        'Model': ['LSTM', 'GRU'],
        'MAE': [mae_lstm, mae_gru],
        'RMSE': [rmse_lstm, rmse_gru],
        'MAPE': [mape_lstm, mape_gru]
    }).sort_values(by='RMSE')
    results.to_csv(f"dl_metrics_{timestamp}.csv", index=False)
    print("\n📈 Comparison:")
    print(results.to_string(index=False))

    # Step 12: Visualization
    plt.figure(figsize=(8,4))
    plt.plot(history_lstm.history['loss'], label='LSTM Train')
    plt.plot(history_lstm.history['val_loss'], label='LSTM Val')
    plt.plot(history_gru.history['loss'], label='GRU Train')
    plt.plot(history_gru.history['val_loss'], label='GRU Val')
    plt.title("Training & Validation Loss")
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(y_true[:200], label="Actual")
    plt.plot(pred_lstm[:200], label="LSTM")
    plt.plot(pred_gru[:200], label="GRU")
    plt.title("Actual vs Predicted (First 200 Samples)")
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    # Step 13: Final Winner
    winner = "LSTM" if rmse_lstm < rmse_gru else "GRU"
    print(f"\n🥇 Final Winner: {winner}\n")
    print("✅ Day 6 Completed.\n")