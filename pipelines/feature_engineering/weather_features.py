
# ============================================================
# 🌦 WEATHER FEATURES PIPELINE (Modular Version)
# ============================================================

import pandas as pd
import numpy as np
import os

def generate_weather_features():

    # ------------------------------------------------------------
    # 1️⃣ LOAD CLEANED DATA
    # ------------------------------------------------------------
    if not os.path.exists("cleaned_dataset_phase1.csv"):
        raise FileNotFoundError("❌ File 'cleaned_dataset_phase1.csv' not found.")

    sale_df = pd.read_csv("cleaned_dataset_phase1.csv")
    df = sale_df.copy()

    print("✅ Cleaned dataset loaded successfully.")
    print("📊 Original Shape:", df.shape)

    # ------------------------------------------------------------
    # 2️⃣ SAFETY BACKUP
    # ------------------------------------------------------------
    backup_name = "Backup_Walmart_Sales_before_WeatherFeatures.csv"
    df.to_csv(backup_name, index=False)
    print(f"🧯 Backup created: {backup_name}")

    # ------------------------------------------------------------
    # 3️⃣ REQUIRED COLUMNS CHECK
    # ------------------------------------------------------------
    required_cols = ['Store', 'Dept', 'Weekly_Sales', 'Date']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"❌ Missing required column: {col}")

    # ------------------------------------------------------------
    # 4️⃣ DATE PREPARATION
    # ------------------------------------------------------------
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)

    # ------------------------------------------------------------
    # 5️⃣ RAIN FLAG
    # ------------------------------------------------------------
    if 'Rainfall' in df.columns:
        df['Rainfall'] = pd.to_numeric(df['Rainfall'], errors='coerce').fillna(0)
        df['Rain'] = (df['Rainfall'] > 0).astype(int)
    else:
        df['Rain'] = 0
        print("⚠ Warning: 'Rainfall' column not found — Rain flag set to 0.")

    print("🌧 Rain feature generated successfully.")

    # ------------------------------------------------------------
    # 6️⃣ SAFE SAVE
    # ------------------------------------------------------------
    output_path = os.path.join("data", "features", "weather_features.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"💾 Weather features saved to: {output_path}")



# ------------------------------------------------------------
# 🚀 MAIN EXECUTION BLOCK
# ------------------------------------------------------------
if __name__ == "__main__":
    generate_weather_features()