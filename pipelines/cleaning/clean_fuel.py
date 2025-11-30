
from utils.file_manager import load_csv, save_csv

def clean_fuel():
    print("[START] Cleaning fuel prices...")
    df = load_csv("data/processed/cleaned_fuel.csv")

    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)

    save_csv(df, "data/processed/cleaned_fuel.csv")
    print("[OK] cleaned_fuel.csv updated")

if __name__ == "__main__":
    clean_fuel()