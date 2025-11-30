
from utils.file_manager import load_csv, save_csv

def clean_social_sentiment():
    print("[START] Cleaning social media sentiment...")
    df = load_csv("data/processed/cleaned_sentiment.csv")

    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)

    save_csv(df, "data/processed/cleaned_sentiment.csv")
    print("[OK] cleaned_sentiment.csv updated")

if __name__ == "__main__":
    clean_social_sentiment()