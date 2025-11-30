# Feature Store – Extracted Features

This folder contains all features extracted from the processed datasets for modeling:

- *Time-based Features:* Day, Week, Month, Season, Holiday flags.
- *Lag & Rolling Features:* Previous sales, moving averages, cumulative sums.
- *Weather Features:* Temperature, Rain, Snow, etc.
- *Trend Features:* Google Trends-based indicators.
- *Synthetic Features:* SMOTE or other data augmentation results.

> *Note:* Features are *generated automatically* by the feature engineering pipeline and are *not included* in the repository.

*Storage & Usage:*  
- Features are saved as CSV or Parquet in data/features/.
- Used directly by model training scripts to ensure consistency and reproducibility.
