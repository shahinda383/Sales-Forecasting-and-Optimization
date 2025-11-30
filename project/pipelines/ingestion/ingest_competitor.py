
from utils.file_manager import load_csv, save_csv

def ingest_competitor():
    print("[START] Ingesting competitor prices...")
    df = load_csv("data/raw/competitor_prices.csv")
    save_csv(df, "data/processed/cleaned_competitor.csv")
    print("[OK] cleaned_competitor.csv saved to data/processed/")

if __name__ == "__main__":
    ingest_competitor()