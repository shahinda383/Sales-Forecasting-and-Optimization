
import pandas as pd
import numpy as np
import os


class TimeFeatureEngineer:

    def _init_(self, input_path="cleaned_dataset_phase1.csv",
                 output_path="data/features/time_features.csv"):

        self.input_path = input_path
        self.output_path = output_path
        self.backup_name = "Backup_Walmart_Sales_before_Features.csv"
        self.stage_file = "Feature_Engineering_Stage1_TimeFeatures.csv"
        self.df = None

    # --------------------------------------------
    # 1️⃣ LOAD DATA SAFELY
    # --------------------------------------------
    def load_data(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"❌ الملف '{self.input_path}' مش موجود.")

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
    # 3️⃣ DATE HANDLING & SORTING
    # --------------------------------------------
    def prepare_dates(self):
        if "Date" not in self.df.columns:
            raise KeyError("❌ 'Date' column missing.")

        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        self.df = self.df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

        print("✅ Data sorted chronologically and Date column parsed successfully.")

    # --------------------------------------------
    # 4️⃣ TIME-BASED FEATURES
    # --------------------------------------------
    def add_time_features(self):

        df = self.df

        df["Day"] = df["Date"].dt.day
        df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Fall"

        df["Season"] = df["Month"].apply(get_season)

        if "IsHoliday" in df.columns:
            df["IsHolidayFlag"] = df["IsHoliday"].astype(int)
        else:
            df["IsHolidayFlag"] = 0
            print("⚠ 'IsHoliday' not found. Defaulted all to 0.")

        print("✅ Time-based features added successfully.")
        self.df = df

    # --------------------------------------------
    # 5️⃣ VALIDATION CHECK
    # --------------------------------------------
    def validate(self):
        df = self.df
        print("\n🔍 Validation Summary:")
        print("=========================")
        print(f"✅ Total Columns Now: {len(df.columns)}")
        print(f"✅ Shape after feature addition: {df.shape}")
        print(f"🧩 Sample:\n{df[['Date','Month','Season','IsWeekend','IsHolidayFlag']].head()}")

    # --------------------------------------------
    # 6️⃣ SAFE SAVE
    # --------------------------------------------
    def save_features(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        self.df.to_csv(self.output_path, index=False)
        print(f"\n💾 Time Features saved to: '{self.output_path}'")

        # Stage 1 save (same as original notebook)
        self.df.to_csv(self.stage_file, index=False)
        print(f"💾 Stage 1 saved as: '{self.stage_file}'")

    # --------------------------------------------
    # 7️⃣ CONFIRM SAVE
    # --------------------------------------------
    def verify_save(self):
        df_check = pd.read_csv(self.stage_file)

        if df_check.shape[1] < self.df.shape[1]:
            print("🚨 Warning: Column mismatch after saving!")
        else:
            print("🎯 Save verified: all columns preserved successfully.")

    # --------------------------------------------
    # RUN FULL PIPELINE
    # --------------------------------------------
    def run(self):
        self.load_data()
        self.backup()
        self.prepare_dates()
        self.add_time_features()
        self.validate()
        self.save_features()
        self.verify_save()


# ============================================================
# 🏁 RUN PIPELINE (in the same file)
# ============================================================
if __name__ == "__main__":
    pipeline = TimeFeatureEngineer(
        input_path="cleaned_dataset_phase1.csv",
        output_path="data/features/time_features.csv"
    )
    pipeline.run()