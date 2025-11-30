
from utils.file_manager import load_csv, save_csv

def ingest_trends():
    print("[START] Ingesting google trends...")
    df = load_csv("data/raw/google_trends.csv")
    save_csv(df, "data/processed/cleaned_trends.csv")
    print("[OK] cleaned_trends.csv saved to data/processed/")

if __name__ == "__main__":
    ingest_trends()