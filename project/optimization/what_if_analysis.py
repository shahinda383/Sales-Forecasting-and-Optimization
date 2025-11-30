# ============================================================
# 💡 What-if Scenario Analysis & Sensitivity Modeling - Modular Pipeline
# Project: Sales Forecasting & Optimization System
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1️⃣ Load Dataset
# ============================================================
def load_data(file_path="Merged_Sales_Data.csv", required_cols=None):
    if required_cols is None:
        required_cols = ['Actual_Sales', 'Pred_final_predictions', 'Sales', 'Size']

    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    print("📂 Columns Detected:", list(data.columns))

    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"❌ Missing columns: {missing}")

    # Base revenue, cost, profit
    data['Revenue'] = data['Actual_Sales'] * 1.0
    data['Cost'] = data['Revenue'] * 0.6
    data['Profit'] = data['Revenue'] - data['Cost']

    base_revenue = data['Revenue'].sum()
    base_profit = data['Profit'].sum()
    print(f"💰 Base Revenue: {base_revenue:,.2f}")
    print(f"💸 Base Profit: {base_profit:,.2f}")

    return data, base_revenue, base_profit


# ============================================================
# 2️⃣ Scenario Simulation Functions
# ============================================================
def simulate_discount(data, discount_percent):
    scenario = data.copy()
    scenario['Discount'] = discount_percent
    scenario['New_Price'] = scenario['Actual_Sales'] * (1 - discount_percent / 100)
    scenario['Sales_Change_Factor'] = 1 + (discount_percent / 50)
    scenario['New_Revenue'] = scenario['New_Price'] * scenario['Sales_Change_Factor']
    scenario['New_Cost'] = scenario['New_Revenue'] * 0.6
    scenario['New_Profit'] = scenario['New_Revenue'] - scenario['New_Cost']
    return scenario

def simulate_stock_increase(data, increase_percent):
    scenario = data.copy()
    scenario['Stock_Increase'] = increase_percent
    scenario['Sales_Change_Factor'] = 1 + (increase_percent / 100) * 0.4
    scenario['New_Revenue'] = scenario['Revenue'] * scenario['Sales_Change_Factor']
    scenario['New_Cost'] = scenario['New_Revenue'] * 0.6
    scenario['New_Profit'] = scenario['New_Revenue'] - scenario['New_Cost']
    return scenario

def combine_scenarios(data, discount, stock):
    scenario = simulate_discount(data, discount)
    scenario = simulate_stock_increase(scenario, stock)
    scenario['Scenario'] = f"Discount {discount}% + Stock {stock}%"
    return scenario


# ============================================================
# 3️⃣ Generate & Summarize Scenarios
# ============================================================
def generate_scenarios(data, base_revenue, base_profit):
    discount_10 = simulate_discount(data, 10)
    stock_20 = simulate_stock_increase(data, 20)
    combo = combine_scenarios(data, 10, 20)

    summary = pd.DataFrame({
        'Scenario': ['Base', 'Discount -10%', 'Stock +20%', 'Combined'],
        'Total_Revenue': [
            base_revenue,
            discount_10['New_Revenue'].sum(),
            stock_20['New_Revenue'].sum(),
            combo['New_Revenue'].sum()
        ],
        'Total_Profit': [
            base_profit,
            discount_10['New_Profit'].sum(),
            stock_20['New_Profit'].sum(),
            combo['New_Profit'].sum()
        ]
    })

    summary['Revenue_Change_%'] = ((summary['Total_Revenue'] - base_revenue) / base_revenue * 100).round(2)
    summary['Profit_Change_%'] = ((summary['Total_Profit'] - base_profit) / base_profit * 100).round(2)

    return summary, discount_10, stock_20, combo


# ============================================================
# 4️⃣ Visualization
# ============================================================
def visualize_scenarios(summary):
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=summary.melt(id_vars='Scenario', value_vars=['Total_Revenue','Total_Profit']),
        x='Scenario', y='value', hue='variable'
    )
    plt.title("💡 Scenario Comparison: Revenue & Profit", fontsize=14, weight='bold')
    plt.ylabel("Value")
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(8,5))
    sns.heatmap(summary[['Revenue_Change_%','Profit_Change_%']].set_index(summary['Scenario']),
                annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("🔥 Sensitivity of Scenarios (%)")
    plt.show()


# ============================================================
# 5️⃣ Save Results
# ============================================================
def save_results(summary, file_name="WhatIf_Scenario_Results.csv"):
    summary.to_csv(file_name, index=False)
    print(f"\n✅ Scenario Analysis saved as '{file_name}'")


# ============================================================
# 6️⃣ Insights Summary
# ============================================================
def print_insights(summary):
    print("\n📈 Strategic Insights:")
    for _, row in summary.iterrows():
        print(f"➡ {row['Scenario']}: Revenue Change {row['Revenue_Change_%']}%, Profit Change {row['Profit_Change_%']}%")

    print("\n🏆 Recommendation:")
    best = summary.loc[summary['Total_Profit'].idxmax()]
    print(f"✔ Best Scenario: {best['Scenario']} with profit increase of {best['Profit_Change_%']}%")


# ============================================================
# 7️⃣ Main Execution
# ============================================================
if __name__ == "__main__":
    data, base_revenue, base_profit = load_data()
    summary, discount_10, stock_20, combo = generate_scenarios(data, base_revenue, base_profit)
    visualize_scenarios(summary)
    save_results(summary)
    print_insights(summary)