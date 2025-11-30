
# ============================================================
# 📦 ROLLING FEATURES PIPELINE (Modular Version)
# ============================================================

import pandas as pd
import numpy as np
import os

def generate_rolling_features():
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
    backup_name = "Backup_Walmart_Sales_before_RollingFeatures.csv"
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
    # 4️⃣ DATE HANDLING
    # ------------------------------------------------------------
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)

    group_key = ['Store', 'Dept']

    # ------------------------------------------------------------
    # 5️⃣ ROLLING AVG FOR TEMPERATURE (7 days)
    # ------------------------------------------------------------
    if 'Temperature_x' in df.columns:

        df['Temperature_x'] = pd.to_numeric(df['Temperature_x'], errors='coerce')

        df['Temperature_x'] = df.groupby(group_key)['Temperature_x'] \
                                .transform(lambda x: x.fillna(x.median()))

        df['Temp_avg'] = (
            df.groupby(group_key)['Temperature_x']
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index(level=group_key, drop=True)
        )
    else:
        df['Temp_avg'] = np.nan
        print("⚠ Warning: 'Temperature_x' column not found — Temp_avg filled with NaN.")

    print("✅ Rolling temperature feature created.")

    # ------------------------------------------------------------
    # 6️⃣ SAFE SAVE
    # ------------------------------------------------------------
    output_path = os.path.join("data", "features", "rolling_features.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"💾 Rolling features saved to: {output_path}")



# ------------------------------------------------------------
# 🚀 MAIN EXECUTION BLOCK
# ------------------------------------------------------------
if __name__ == "_main_":
    generate_rolling_features()