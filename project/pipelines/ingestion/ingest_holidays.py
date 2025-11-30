
from utils.file_manager import load_csv, save_csv

def ingest_holidays():
    print("[START] Ingesting holidays...")
    df = load_csv("data/raw/public_holidays.csv")
    save_csv(df, "data/processed/cleaned_holidays.csv")
    print("[OK] cleaned_holidays.csv saved to data/processed/")

if __name__ == "__main__":
    ingest_holidays()