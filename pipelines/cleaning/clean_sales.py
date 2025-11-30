
import pandas as pd
from utils.file_manager import load_csv, save_csv

def clean_sales():
    print("[START] Cleaning sales data...")
    df = load_csv("data/processed/cleaned_sales.csv")

    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    save_csv(df, "data/processed/cleaned_sales.csv")
    print("[OK] cleaned_sales.csv updated")

if __name__ == "__main__":
    clean_sales()