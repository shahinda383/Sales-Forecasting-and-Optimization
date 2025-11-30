
from utils.file_manager import load_csv, save_csv

def ingest_weather():
    print("[START] Ingesting weather data...")
    df = load_csv("data/raw/weather_data.csv")
    save_csv(df, "data/processed/cleaned_weather.csv")
    print("[OK] cleaned_weather.csv saved to data/processed/")

if __name__ == "__main__":
    ingest_weather()