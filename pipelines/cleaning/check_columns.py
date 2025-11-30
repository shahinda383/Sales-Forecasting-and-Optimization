
import pandas as pd
from pathlib import Path

PROC = Path("data/processed")

files = [
    "cleaned_sales.csv",
    "cleaned_weather.csv",
    "cleaned_trends.csv",
    "cleaned_holidays.csv",
    "cleaned_sentiment.csv",
    "cleaned_macro.csv",
    "cleaned_fuel.csv",
    "cleaned_competitor.csv"
]

for f in files:
    print("\n===========================")
    print("Columns in:", f)
    try:
        df = pd.read_csv(PROC / f)
        print(df.columns.tolist())
    except Exception as e:
        print("ERROR:", e)