
from utils.file_manager import load_csv, save_csv

def ingest_macro():
    print("[START] Ingesting macroeconomic data...")
    df = load_csv("data/raw/macroeconomic_data.csv")
    save_csv(df, "data/processed/cleaned_macro.csv")
    print("[OK] cleaned_macro.csv saved to data/processed/")

if __name__ == "__main__":
    ingest_macro()