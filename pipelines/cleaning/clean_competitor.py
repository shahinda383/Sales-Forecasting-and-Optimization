
from utils.file_manager import load_csv, save_csv

def clean_competitor():
    print("[START] Cleaning competitor prices...")
    df = load_csv("data/processed/cleaned_competitor.csv")

    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)

    save_csv(df, "data/processed/cleaned_competitor.csv")
    print("[OK] cleaned_competitor.csv updated")

if __name__ == "__main__":
    clean_competitor()