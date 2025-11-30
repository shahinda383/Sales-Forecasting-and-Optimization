
import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURATIONS
# ============================================================
DROP_IF_MISSING_RATIO_ABOVE = 0.7
WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99
MARKDOWN_FILL_VALUE = 0


# ============================================================
# MAIN FUNCTION
# ============================================================
def explore_data(input_path="data/processed/merged_dataset.csv", output_path="data/processed/cleaned_dataset_phase1.csv"):

    print("\n🚀 Starting Explore & Clean Pipeline...\n")

    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ File not found: {input_path}")

    # استخدام low_memory=False لتجنب DtypeWarning
    df = pd.read_csv(input_path, low_memory=False) 
    print(f"📥 Loaded dataset → {df.shape} rows, {len(df.columns)} columns\n")

    # Ensure date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        print("✅ Date column converted to datetime.\n")
    else:
        print("⚠ Warning: 'Date' column not found.\n")

    # --------------------------------------------------------
    # INITIAL CLEANING
    # --------------------------------------------------------
    print("🚿 Starting comprehensive cleaning process...\n")

    # Drop columns with >70% missing
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > DROP_IF_MISSING_RATIO_ABOVE].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore") 
    print(f"🧹 Dropped {len(cols_to_drop)} columns with >70% missing values.\n")

    # Drop constant columns
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df.drop(columns=constant_cols, inplace=True, errors="ignore") 
    print(f"🗑 Dropped {len(constant_cols)} constant columns.\n")

    # Remove duplicates
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"🧩 Removed {dup_count} duplicate rows.\n")

    # --------------------------------------------------------
    # HANDLE DATA TYPES
    # --------------------------------------------------------
    cat_candidates = ['Type_x', 'City', 'Holiday', 'Type_y']
    for c in cat_candidates:
        if c in df.columns:
            df[c] = df[c].astype('category')

    # Convert numeric-like objects
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # --------------------------------------------------------
    # HANDLE MISSING VALUES
    # --------------------------------------------------------

    # Fill MarkDown columns
    markdown_cols = [c for c in df.columns if 'markdown' in c.lower() or 'MarkDown' in c]
    for col in markdown_cols:
        df[col] = df[col].fillna(MARKDOWN_FILL_VALUE)

    # Fill key economic indicators
    for col in ['CPI', 'Fuel_Price_x', 'Unemployment']:
        if col in df.columns:
            med = df[col].median(skipna=True) 
            df[col].fillna(med, inplace=True) 
            print(f"💰 Filled {col} NaNs with median: {med:.2f}")

    # Fill lag/rolling columns
    lag_rolling_cols = [c for c in df.columns if any(x in c for x in
                        ['Lag', 'Rolling', 'Expanding', 'PctChange', 'Volatility'])]
    for col in lag_rolling_cols:
        # Forward fill, then backward fill to cover gaps
        df[col] = df[col].ffill().bfill()

    # Fill decomposition columns
    decomp_cols = [c for c in df.columns if 'Decomp' in c]
    for col in decomp_cols:
        med = df[col].median(skipna=True)
        df[col].fillna(med, inplace=True)
        print(f"🧠 Filled {col} NaNs with median: {med:.2f}")

    # Handle categorical NaNs
    if "Holiday" in df.columns:
        # Add category first, then fill
        df["Holiday"] = df["Holiday"].cat.add_categories(["No Holiday"]).fillna("No Holiday")

    if "Type_y" in df.columns:
        # Add category first, then fill
        df["Type_y"] = df["Type_y"].cat.add_categories(["Unknown"]).fillna("Unknown")

    # --------------------------------------------------------
    # OUTLIER HANDLING (Winsorization)
    # --------------------------------------------------------
    if "Weekly_Sales" in df.columns:
        low_q = df["Weekly_Sales"].quantile(WINSORIZE_LOWER)
        high_q = df["Weekly_Sales"].quantile(WINSORIZE_UPPER)
        # Using clip() for Winsorization based on quantiles
        df["Weekly_Sales_wins"] = df["Weekly_Sales"].clip(lower=low_q, upper=high_q)
        print(f"✅ Winsorized Weekly_Sales between [{low_q:.2f}, {high_q:.2f}]")
    else:
        print("⚠ 'Weekly_Sales' not found! Skipping Winsorization.")

    # --------------------------------------------------------
    # HANDLE INFINITE VALUES
    # --------------------------------------------------------
    # Count infinites only in numeric columns
    inf_count_before = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"🔁 Replaced {inf_count_before} infinite values with NaN.\n")

    # --------------------------------------------------------
    # FINAL FILL OF REMAINING NaNs
    # --------------------------------------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().sum() > 0:
            med = df[col].median(skipna=True)
            df[col].fillna(med, inplace=True)

    # Downcast numerics
    df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast='float')

    # --------------------------------------------------------
    # VALIDATION + SAVE
    # --------------------------------------------------------
    inf_count_after = np.isinf(df[num_cols]).sum().sum()
    nan_count = df.isna().sum().sum()

    print("\n🔍 FINAL VALIDATION REPORT")
    print("==========================")
    print(f"Remaining ∞ values: {inf_count_after}")
    print(f"Remaining NaN values: {nan_count}")
    print(f"Final Dataset Shape: {df.shape}")
    print(f"Total Columns: {len(df.columns)}")

    df.to_csv(output_path, index=False) 
    print(f"\n💾 Cleaned dataset saved → {output_path}")

    print("\n🎯 Dataset CLEAN & READY! ✔\n")

    return df


# ============================================================
# Run as Script
# ============================================================
if __name__ == "__main__":
    explore_data()