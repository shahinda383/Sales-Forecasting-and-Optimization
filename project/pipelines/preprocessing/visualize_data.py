
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# CONFIGURATIONS
# ============================================================
INPUT_PATH = "data/processed/cleaned_dataset_phase1.csv"

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def visualize_data(input_path=INPUT_PATH):

    print("\n🚀 Script started successfully!")
    print(f"🔎 Looking for file: {input_path}")
    
    if not os.path.exists(input_path):
        print("❌ File not found!!!!")
        raise FileNotFoundError(f"File not found: {input_path}")
    
    # --------------------------------------------------------
    # Load Data and Setup
    # --------------------------------------------------------
    sale_df = pd.read_csv(input_path, low_memory=False)
    print("✅ Data loaded successfully!")

    # Ensure Date column is in datetime format
    if "Date" in sale_df.columns:
        sale_df["Date"] = pd.to_datetime(sale_df["Date"], errors='coerce')
        sale_df = sale_df.sort_values('Date').reset_index(drop=True)
        print("✅ Date column sorted and ready for time-series plotting.\n")
    else:
        print("⚠ Warning: 'Date' column not found or converted successfully.\n")
    
    # Set style for better visualizations
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # --------------------------------------------------------
    # 1. Weekly Sales Over Time
    # --------------------------------------------------------
    sales_col = 'Weekly_Sales_wins' if 'Weekly_Sales_wins' in sale_df.columns else 'Weekly_Sales'

    if "Weekly_Sales" in sale_df.columns and "Date" in sale_df.columns:
        print("🎨 Plotting 1/3: Weekly Sales Trend.")
        plt.figure(figsize=(15, 6))
        sns.lineplot(data=sale_df, x='Date', y=sales_col)
        plt.title('Weekly Sales Over Time (Overall Trend)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(f'Weekly Sales ({sales_col})', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠ Skipping Weekly Sales Trend plot: 'Weekly_Sales' or 'Date' missing.")
    
    # --------------------------------------------------------
    # 2. Average Sales: Holidays vs Non-Holidays
    # --------------------------------------------------------
    if "Weekly_Sales" in sale_df.columns and "Holiday" in sale_df.columns:
        print("🎨 Plotting 2/3: Holiday Impact.")
        plt.figure(figsize=(8, 5))
        holiday_sales = sale_df.groupby(sale_df['Holiday'] != 'No Holiday')[sales_col].mean().reset_index()
        holiday_sales['Holiday'] = holiday_sales['Holiday'].replace({True: 'Holiday', False: 'No Holiday'})
        sns.barplot(data=holiday_sales, x='Holiday', y=sales_col, palette='viridis', errorbar=None)
        plt.title('Average Weekly Sales: Holidays vs Non-Holidays', fontsize=16)
        plt.xlabel('Is Holiday', fontsize=12)
        plt.ylabel(f'Average Weekly Sales ({sales_col})', fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠ Skipping Holiday Impact plot: 'Weekly_Sales' or 'Holiday' missing.")
    
    # --------------------------------------------------------
    # 3. Weather Variables vs Weekly Sales
    # --------------------------------------------------------
    weather_cols = ['Temperature_x', 'Rainfall', 'Humidity']
    if "Weekly_Sales" in sale_df.columns:
        print("🎨 Plotting 3/3: Weather Variables Relationship.")
        fig, axes = plt.subplots(1, len(weather_cols), figsize=(18, 5))
        for i, col in enumerate(weather_cols):
            ax = axes[i]
            if col in sale_df.columns:
                sns.scatterplot(data=sale_df, x=col, y=sales_col, ax=ax, alpha=0.6)
                ax.set_title(f'{col} vs Weekly Sales', fontsize=12)
                ax.set_xlabel(col)
                ax.set_ylabel('Weekly Sales')
            else:
                ax.text(0.5, 0.5, f'No {col} column', ha='center', va='center', fontsize=12, color='red')
                ax.set_title(f'{col} vs Weekly Sales')
                ax.set_xticks([])
                ax.set_yticks([])
        plt.tight_layout()
        plt.show()
    else:
        print("⚠ Skipping Weather Plots: 'Weekly_Sales' missing.")
    
    print("\n✅ Visualization complete! ✔\n")

# ============================================================
# Run as Script
# ============================================================
if __name__ == "__main__":
    visualize_data()