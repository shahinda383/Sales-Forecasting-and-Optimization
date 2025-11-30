
from utils.file_manager import load_csv, save_csv

def clean_trends():
    print("[START] Cleaning trends data...")
    df = load_csv("data/processed/cleaned_trends.csv")
    df.fillna(0, inplace=True)
    save_csv(df, "data/processed/cleaned_trends.csv")
    print("[OK] cleaned_trends.csv updated")

if __name__ == "__main__":
    clean_trends()