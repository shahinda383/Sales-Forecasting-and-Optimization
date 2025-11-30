
from utils.file_manager import load_csv, save_csv

def ingest_social_sentiment():
    print("[START] Ingesting social media sentiment...")
    df = load_csv("data/raw/social_media_sentiment.csv")
    save_csv(df, "data/processed/cleaned_sentiment.csv")
    print("[OK] cleaned_sentiment.csv saved to data/processed/")

if __name__ == "__main__":
    ingest_social_sentiment()