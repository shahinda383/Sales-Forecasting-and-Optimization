
import pandas as pd
import numpy as np
import os


class LagFeatureEngineer:

    def _init_(self, input_path="cleaned_dataset_phase1.csv",
                 output_path="data/features/lag_features.csv"):

        self.input_path = input_path
        self.output_path = output_path
        self.backup_name = "Backup_Walmart_Sales_before_LagFeatures.csv"
        self.df = None

    # --------------------------------------------
    # 1️⃣ LOAD DATA SAFELY
    # --------------------------------------------
    def load_data(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"❌ File '{self.input_path}' not found.")

        sale_df = pd.read_csv(self.input_path)
        self.df = sale_df.copy()

        print("✅ Cleaned dataset loaded successfully.")
        print("📊 Original Shape:", self.df.shape)

    # --------------------------------------------
    # 2️⃣ SAFETY BACKUP
    # --------------------------------------------
    def backup(self):
        self.df.to_csv(self.backup_name, index=False)
        print(f"🧯 Backup created: {self.backup_name}")

    # --------------------------------------------
    # 3️⃣ VALIDATE REQUIRED COLUMNS
    # --------------------------------------------
    def validate_columns(self):

        required_cols = ['Store', 'Dept', 'Weekly_Sales', 'Date']
        for col in required_cols:
            if col not in self.df.columns:
                raise KeyError(f"❌ Missing column: {col}")

        print("✅ Required columns verified successfully.")

    # --------------------------------------------
    # 4️⃣ DATE HANDLING & SORTING
    # --------------------------------------------
    def prepare_dates(self):

        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        self.df = self.df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)

        print("✅ Date parsed and data sorted correctly.")

    # --------------------------------------------
    # 5️⃣ CREATE LAG FEATURES
    # --------------------------------------------
    def create_lag_features(self):

        df = self.df
        group_key = ['Store', 'Dept']

        # Previous week sales
        df['Sales_prev_week'] = df.groupby(group_key)['Weekly_Sales'].shift(1)

        # Smart filling
        median_by_group = df.groupby(group_key)['Weekly_Sales'].transform('median')
        overall_median = df['Weekly_Sales'].median()

        df['Sales_prev_week'] = df['Sales_prev_week'].fillna(median_by_group)
        df['Sales_prev_week'] = df['Sales_prev_week'].fillna(overall_median)

        # Validation
        if df['Sales_prev_week'].isna().sum() == 0:
            print("✅ Lag feature created successfully and NaNs handled.")
        else:
            print("⚠ Warning: There are still NaN values!")

        print("\n🧪 Sample Preview:")
        print(df[['Store', 'Dept', 'Date', 'Weekly_Sales', 'Sales_prev_week']].head(10))

        self.df = df

    # --------------------------------------------
    # 6️⃣ SAFE SAVE
    # --------------------------------------------
    def save_features(self):

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        self.df.to_csv(self.output_path, index=False)
        print(f"\n💾 Lag Features saved to: '{self.output_path}'")

    # --------------------------------------------
    # RUN FULL PIPELINE
    # --------------------------------------------
    def run(self):
        self.load_data()
        self.backup()
        self.validate_columns()
        self.prepare_dates()
        self.create_lag_features()
        self.save_features()


# ============================================================
# 🏁 RUN PIPELINE (in the same file)
# ============================================================
if __name__ == "__main__":
    pipeline = LagFeatureEngineer(
        input_path="cleaned_dataset_phase1.csv",
        output_path="data/features/lag_features.csv"
    )
    pipeline.run()