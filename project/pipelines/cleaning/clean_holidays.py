
from utils.file_manager import load_csv, save_csv

def clean_holidays():
    print("[START] Cleaning holidays...")
    df = load_csv("data/processed/cleaned_holidays.csv")
    df.drop_duplicates(inplace=True)
    save_csv(df, "data/processed/cleaned_holidays.csv")
    print("[OK] cleaned_holidays.csv updated")

if __name__ == "__main__":
    clean_holidays()