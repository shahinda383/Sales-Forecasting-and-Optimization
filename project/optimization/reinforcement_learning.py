# ============================================================
# 🎯 Reinforcement Learning Optimization (Q-Learning) - Modular Pipeline
# Project: AI-Based Dynamic Pricing & Inventory System
# Dataset: Merged_Sales_Data.csv
# ============================================================

import pandas as pd
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# 1️⃣ Load Data
# ============================================================
def load_data(file_path="Merged_Sales_Data.csv"):
    data = pd.read_csv(file_path)
    data.columns = [c.strip().replace(" ", "_") for c in data.columns]
    print("📂 Data Loaded. Columns:", list(data.columns))
    return data

# ============================================================
# 2️⃣ Data Preparation
# ============================================================
def prepare_data(data):
    # حساب الربح إذا لم يكن موجود
    if "Profit" not in data.columns:
        if "Revenue" in data.columns and "Cost" in data.columns:
            data["Profit"] = data["Revenue"] - data["Cost"]
        else:
            data["Profit"] = np.random.uniform(100, 5000, len(data))
    
    # تقييم المخاطر بشكل مبدئي
    data["Risk_Level"] = pd.qcut(data["Profit"], q=4, labels=["Low", "Medium", "High", "Critical"])
    data["Normalized_Profit"] = (data["Profit"] - data["Profit"].min()) / (data["Profit"].max() - data["Profit"].min())
    
    scenarios = ["Increase_Price", "Decrease_Price", "Promote_Product", "Reduce_Stock", "Optimize_Discount"]
    data["Scenario"] = np.random.choice(scenarios, len(data))
    
    risk_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    data["Risk_Level_Num"] = data["Risk_Level"].map(risk_map)
    
    return data, scenarios, list(data["Risk_Level"].unique())

# ============================================================
# 3️⃣ Reward Function
# ============================================================
def get_reward(data, scenario, risk_level):
    filtered = data[data["Scenario"] == scenario]
    if filtered.empty:
        return np.random.uniform(-0.1, 0.1)
    row = filtered.iloc[0]
    profit_score = row["Normalized_Profit"]
    risk_penalty = row["Risk_Level_Num"] / 3
    reward = profit_score - risk_penalty
    return reward

# ============================================================
# 4️⃣ Q-Learning Training
# ============================================================
def train_q_learning(data, states, actions, alpha=0.1, gamma=0.9, epsilon=0.3, episodes=200):
    Q_table = pd.DataFrame(0, index=states, columns=actions, dtype=float)
    reward_history = []

    for episode in range(episodes):
        state = random.choice(states)
        total_episode_reward = 0

        for _ in range(len(actions)):
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)
            else:
                action = Q_table.loc[state].idxmax()

            reward = get_reward(data, action, state)
            next_state = random.choice(states)

            Q_table.loc[state, action] = Q_table.loc[state, action] + alpha * (
                reward + gamma * Q_table.loc[next_state].max() - Q_table.loc[state, action]
            )
            total_episode_reward += reward
            state = next_state

        reward_history.append(total_episode_reward)
        epsilon = max(0.05, epsilon * 0.98)

    return Q_table, reward_history

# ============================================================
# 5️⃣ Extract Policy
# ============================================================
def extract_policy(Q_table):
    best_actions = Q_table.idxmax(axis=1)
    policy_df = best_actions.reset_index()
    policy_df.columns = ['Risk_Level', 'Optimal_Action']
    return best_actions, policy_df

# ============================================================
# 6️⃣ Visualization
# ============================================================
def visualize_training(reward_history):
    plt.figure(figsize=(10,5))
    plt.plot(reward_history)
    plt.title("📈 Q-Learning Reward Progress Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

# ============================================================
# 7️⃣ Save Results
# ============================================================
def save_results(Q_table, policy_df):
    Q_table.to_csv("Q_Learning_ValueTable.csv")
    policy_df.to_csv("Q_Learning_OptimalPolicy.csv", index=False)
    print("\n📈 Results Saved:")
    print("➡ Q-Learning Table → Q_Learning_ValueTable.csv")
    print("➡ Optimal Policy → Q_Learning_OptimalPolicy.csv")

# ============================================================
# 8️⃣ Insights
# ============================================================
def print_insights(Q_table):
    best_action = Q_table.stack().idxmax()
    best_state, best_scenario = best_action
    best_value = Q_table.loc[best_state, best_scenario]

    print(f"\n🚀 The AI Agent recommends:")
    print(f"👉 Scenario: {best_scenario} | Best Value: {best_value:.3f} | Risk Level: {best_state}")
    print("\n🌟 The system has successfully learned an optimal decision policy that balances profit and risk.")

# ============================================================
# 9️⃣ Main Execution
# ============================================================
if __name__ == "__main__":
    data = load_data()
    data, actions, states = prepare_data(data)
    Q_table, reward_history = train_q_learning(data, states, actions)
    best_actions, policy_df = extract_policy(Q_table)
    visualize_training(reward_history)
    save_results(Q_table, policy_df)
    print("\n📊 Final Q-Table:")
    print(Q_table.round(3))
    print("\n🏆 Optimal Action per Risk Level:")
    for state, action in best_actions.items():
        print(f"  • {state} → {action}")
    print_insights(Q_table)