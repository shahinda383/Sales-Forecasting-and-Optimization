# ============================================================
# 🌍 Advanced Supply Chain Optimization - Modular Pipeline
# Project: Sales Forecasting & Optimization System
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pulp
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1️⃣ Load & Prepare Data
# ============================================================
def load_prepare_data(file_path="Day2_Inventory_Optimization_Report.csv"):
    data = pd.read_csv(file_path)
    print("✅ Data Loaded Successfully | Shape:", data.shape)
    print("📋 Columns:", data.columns.tolist())
    
    for col in ['Optimal_Stock', 'Price', 'Cost', 'Expected_Profit', 'Store_Size']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    data['Dept'] = data['Dept'].astype(str)
    
    # Simulate supply chain factors
    np.random.seed(42)
    data['Supplier_LeadTime'] = np.random.randint(1, 10, len(data))
    data['Demand_Variability'] = np.random.uniform(0.85, 1.15, len(data))
    data['Available_Stock'] = data['Optimal_Stock'] * np.random.uniform(0.8, 1.2, len(data))
    data['Adjusted_Demand'] = data['Optimal_Stock'] * data['Demand_Variability']
    data['Supplier_Reliability'] = np.random.uniform(0.85, 0.99, len(data))
    data['Max_Capacity'] = data['Store_Size'] * 1.5
    
    return data

# ============================================================
# 2️⃣ Define & Solve LP Model
# ============================================================
def solve_supply_chain_lp(data, total_stock_limit=150000):
    model = pulp.LpProblem("SupplyChain_Optimization", pulp.LpMaximize)
    departments = data['Dept'].tolist()
    
    order_qty = pulp.LpVariable.dicts("Order_Qty", departments, lowBound=0)
    
    model += pulp.lpSum([
        ((data.loc[i, 'Price'] - data.loc[i, 'Cost']) * 
         (data.loc[i, 'Available_Stock'] + order_qty[d]) * 
         data.loc[i, 'Supplier_Reliability']) - 
        (0.02 * data.loc[i, 'Supplier_LeadTime'] * order_qty[d])
        for i, d in enumerate(departments)
    ]), "Total_Optimized_Profit"
    
    # Constraints
    model += pulp.lpSum([order_qty[d] for d in departments]) <= total_stock_limit, "TotalStockLimit"
    
    for i, d in enumerate(departments):
        model += data.loc[i, 'Available_Stock'] + order_qty[d] <= data.loc[i, 'Max_Capacity'], f"Capacity_{d}"
        model += data.loc[i, 'Available_Stock'] + order_qty[d] >= 0.9 * data.loc[i, 'Adjusted_Demand'], f"Demand_{d}"
    
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    print("✅ Optimization Solved Successfully with Constraints")
    return model, order_qty, departments

# ============================================================
# 3️⃣ Collect Results
# ============================================================
def collect_results(data, order_qty, departments):
    results = []
    for i, d in enumerate(departments):
        order = pulp.value(order_qty[d])
        total_stock = data.loc[i, 'Available_Stock'] + order
        profit = ((data.loc[i, 'Price'] - data.loc[i, 'Cost']) * total_stock *
                  data.loc[i, 'Supplier_Reliability']) - (0.02 * data.loc[i, 'Supplier_LeadTime'] * order)
        
        results.append({
            'Dept': d,
            'Order_Qty': round(order, 2),
            'Available_Stock': round(data.loc[i, 'Available_Stock'], 2),
            'Adjusted_Demand': round(data.loc[i, 'Adjusted_Demand'], 2),
            'LeadTime_Days': int(data.loc[i, 'Supplier_LeadTime']),
            'Supplier_Reliability': round(data.loc[i, 'Supplier_Reliability'], 2),
            'Expected_Profit': round(profit, 2),
            'Capacity_Limit': round(data.loc[i, 'Max_Capacity'], 2)
        })
    
    results_df = pd.DataFrame(results)
    results_df['Fulfillment_Rate'] = (results_df['Available_Stock'] + results_df['Order_Qty']) / results_df['Adjusted_Demand']
    results_df['Profit_Rank'] = results_df['Expected_Profit'].rank(ascending=False)
    
    # Advanced Analytics
    results_df['Risk_Adjusted_Profit'] = results_df['Expected_Profit'] * results_df['Supplier_Reliability']
    results_df['Stock_Risk_Index'] = np.abs(results_df['Adjusted_Demand'] - results_df['Available_Stock']) / results_df['Adjusted_Demand']
    results_df['Performance_Score'] = (
        0.5 * results_df['Fulfillment_Rate'] +
        0.3 * results_df['Supplier_Reliability'] +
        0.2 * (1 - results_df['Stock_Risk_Index'])
    )
    
    results_df = results_df.sort_values('Performance_Score', ascending=False)
    return results_df

# ============================================================
# 4️⃣ Visualization
# ============================================================
def visualize_results(results_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Dept', y='Expected_Profit', data=results_df)
    plt.title("💰 Optimized Profit After Supply Chain Constraints")
    plt.tight_layout()
    plt.show()
    
    px.scatter(results_df,
               x='Supplier_Reliability', y='Fulfillment_Rate',
               size='Expected_Profit', color='Dept',
               title='🌍 Supply Reliability vs Demand Fulfillment',
               hover_data=['LeadTime_Days', 'Performance_Score']).show()

# ============================================================
# 5️⃣ Save Results
# ============================================================
def save_results(results_df, file_path="Day3_SupplyChain_Optimization_Report.csv"):
    results_df.to_csv(file_path, index=False)
    print(f"📦 Saved: {file_path}")

# ============================================================
# 6️⃣ Summary Metrics
# ============================================================
def print_summary(results_df):
    total_profit = results_df['Expected_Profit'].sum()
    top_dept = results_df.loc[results_df['Expected_Profit'].idxmax(), 'Dept']
    avg_fulfillment = results_df['Fulfillment_Rate'].mean()
    
    print(f"\n🏆 Total Optimized Profit (After Constraints): ${total_profit:,.2f}")
    print(f"⭐ Best Performing Dept: {top_dept}")
    print(f"📈 Average Fulfillment Rate: {avg_fulfillment:.2%}")

# ============================================================
# 7️⃣ Main Pipeline Execution
# ============================================================
if __name__ == "__main__":
    data = load_prepare_data()
    model, order_qty, departments = solve_supply_chain_lp(data)
    results_df = collect_results(data, order_qty, departments)
    visualize_results(results_df)
    save_results(results_df)
    print_summary(results_df)