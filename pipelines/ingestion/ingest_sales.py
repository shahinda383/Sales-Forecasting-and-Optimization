
from utils.file_manager import load_csv, save_csv

def ingest_sales():
    print("[START] Ingesting sales data...")

    train = load_csv("data/raw/train.csv")
    stores = load_csv("data/raw/stores.csv")
    features = load_csv("data/raw/features.csv")

    merged = train.merge(stores, on="Store", how="left")
    merged = merged.merge(features, on=["Store", "Date"], how="left")

    save_csv(merged, "data/processed/cleaned_sales.csv")
    print("[OK] cleaned_sales.csv saved to data/processed/")

if __name__ == "__main__":
    ingest_sales()