
from utils.file_manager import load_csv, save_csv

def ingest_fuel():
    print("[START] Ingesting fuel prices...")
    df = load_csv("data/raw/fuel_prices.csv")
    save_csv(df, "data/processed/cleaned_fuel.csv")
    print("[OK] cleaned_fuel.csv saved to data/processed/")

if __name__ == "__main__":
    ingest_fuel()