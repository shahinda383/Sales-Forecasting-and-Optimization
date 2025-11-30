
from utils.file_manager import load_csv, save_csv

def clean_weather():
    print("[START] Cleaning weather data...")
    df = load_csv("data/processed/cleaned_weather.csv")
    df = df.drop_duplicates()
    save_csv(df, "data/processed/cleaned_weather.csv")
    print("[OK] cleaned_weather.csv updated")

if __name__ == "__main__":
    clean_weather()