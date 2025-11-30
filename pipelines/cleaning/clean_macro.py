
from utils.file_manager import load_csv, save_csv

def clean_macro():
    print("[START] Cleaning macroeconomic data...")
    df = load_csv("data/processed/cleaned_macro.csv")

    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)

    save_csv(df, "data/processed/cleaned_macro.csv")
    print("[OK] cleaned_macro.csv updated")

if __name__ == "__main__":
    clean_macro()