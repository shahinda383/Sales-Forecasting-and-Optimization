# ============================================================
# 🎲 Monte Carlo Simulation for Risk Analysis - Modular Pipeline
# Project: Sales Forecasting & Optimization System
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1️⃣ Load Scenario Results
# ============================================================
def load_scenario_results(file_path="WhatIf_Scenario_Results.csv"):
    data = pd.read_csv(file_path)
    print("📂 Loaded Scenarios:")
    print(data, "\n")
    return data

# ============================================================
# 2️⃣ Monte Carlo Simulation Function
# ============================================================
def monte_carlo_simulation(data, num_simulations=10000):
    results = []

    for i, row in data.iterrows():
        scenario = row['Scenario']
        base_profit = row['Total_Profit']

        # تقلبات الأسعار والطلب والتكلفة
        price_fluctuation = np.random.normal(1.0, 0.10, num_simulations)
        demand_fluctuation = np.random.normal(1.0, 0.15, num_simulations)
        cost_fluctuation = np.random.normal(1.0, 0.08, num_simulations)

        simulated_profit = base_profit * (price_fluctuation * demand_fluctuation / cost_fluctuation)

        results.append(pd.DataFrame({
            'Scenario': scenario,
            'Simulated_Profit': simulated_profit
        }))

    simulation_results = pd.concat(results, ignore_index=True)
    return simulation_results

# ============================================================
# 3️⃣ Statistical Analysis of Simulation
# ============================================================
def analyze_simulation(simulation_results):
    summary = simulation_results.groupby("Scenario")['Simulated_Profit'].agg([
        ('Mean_Profit', 'mean'),
        ('Std_Deviation', 'std'),
        ('5th_Percentile', lambda x: np.percentile(x, 5)),
        ('95th_Percentile', lambda x: np.percentile(x, 95)),
        ('Prob_Loss_%', lambda x: np.mean(x < 0) * 100)
    ]).reset_index()

    summary['Risk_Level'] = pd.cut(summary['Prob_Loss_%'],
                                   bins=[0, 5, 15, 30, 100],
                                   labels=['🟢 Low Risk', '🟡 Moderate', '🟠 High', '🔴 Critical'])
    print("📊 Monte Carlo Summary:")
    print(summary, "\n")
    return summary

# ============================================================
# 4️⃣ Visualization
# ============================================================
def visualize_simulation(simulation_results, summary):
    plt.figure(figsize=(12,6))
    sns.histplot(data=simulation_results, x='Simulated_Profit', hue='Scenario', bins=70, kde=True, alpha=0.5)
    plt.title("🎲 Monte Carlo Simulation: Profit Distribution by Scenario", fontsize=14, weight='bold')
    plt.xlabel("Simulated Profit")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(9,5))
    sns.barplot(data=summary, x='Scenario', y='Mean_Profit', hue='Risk_Level')
    plt.title("💰 Average Simulated Profit & Risk Level", fontsize=14, weight='bold')
    plt.ylabel("Mean Profit")
    plt.grid(alpha=0.3)
    plt.show()

# ============================================================
# 5️⃣ Save Results
# ============================================================
def save_simulation_results(summary, file_name="MonteCarlo_Simulation_Results.csv"):
    summary.to_csv(file_name, index=False)
    print(f"✅ Results saved as '{file_name}'")

# ============================================================
# 6️⃣ Insights & Recommendations
# ============================================================
def print_insights(summary):
    print("\n📈 Key Insights:")
    for _, row in summary.iterrows():
        print(f"➡ {row['Scenario']}: Mean Profit = {row['Mean_Profit']:.2f}, Risk = {row['Risk_Level']}, "
              f"Loss Probability = {row['Prob_Loss_%']:.1f}%")

    best_scenario = summary.loc[summary['Mean_Profit'].idxmax()]
    print("\n🏆 Recommended Scenario:")
    print(f"✔ {best_scenario['Scenario']} — Highest Expected Profit ({best_scenario['Mean_Profit']:.2f}) "
          f"with {best_scenario['Risk_Level']} risk level.")

# ============================================================
# 7️⃣ Main Execution
# ============================================================
if __name__ == "__main__":
    data = load_scenario_results()
    simulation_results = monte_carlo_simulation(data)
    summary = analyze_simulation(simulation_results)
    visualize_simulation(simulation_results, summary)
    save_simulation_results(summary)
    print_insights(summary)