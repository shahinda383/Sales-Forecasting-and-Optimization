# ============================================================
# 🚀 PHASE 2: ML Engineering - Day 2
# Baseline Models Pipeline (Linear Regression + Decision Tree)
# ============================================================

# ==============================
# 1. Import Required Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


# ==============================
# 2. Load Data Function
# ==============================
def load_data():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv")
    y_test = pd.read_csv("y_test.csv")

    print("✅ Data loaded successfully!")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("=" * 70)

    return X_train, X_test, y_train, y_test


# ==============================
# 3. Feature Scaling Function
# ==============================
def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("📏 Feature scaling completed (for Linear Regression)")
    print("=" * 70)

    return X_train_scaled, X_test_scaled, scaler


# ==============================
# 4. Train Linear Regression
# ==============================
def train_linear_regression(X_train_scaled, y_train):
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    return lr_model


# ==============================
# 5. Train Decision Tree
# ==============================
def train_decision_tree(X_train, y_train):
    tree_model = DecisionTreeRegressor(
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42
    )
    tree_model.fit(X_train, y_train)
    return tree_model


# ==============================
# 6. Evaluation Function
# ==============================
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R²": r2}


# ==============================
# 7. Visualization Functions
# ==============================
def plot_model_performance(results_df):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=results_df.index, y=results_df["R²"], palette="coolwarm")
    plt.title("Model R² Score Comparison", fontsize=14)
    plt.ylabel("R² Score")
    plt.ylim(0, 1)
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=results_df.index, y=results_df["RMSE"], palette="crest")
    plt.title("Model RMSE Comparison", fontsize=14)
    plt.ylabel("RMSE")
    plt.show()


def plot_feature_importance(tree_model, X_train):
    feat_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": tree_model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(y="Feature", x="Importance", data=feat_importance, palette="viridis")
    plt.title("🌳 Top 15 Important Features (Decision Tree)", fontsize=14)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.show()


def plot_residuals(y_test, y_pred_lr, y_pred_tree):
    residuals_lr = y_test.values.flatten() - y_pred_lr.flatten()
    residuals_tree = y_test.values.flatten() - y_pred_tree.flatten()

    plt.figure(figsize=(10, 5))
    sns.histplot(residuals_lr, bins=50, kde=True, color="skyblue", label="Linear Regression")
    sns.histplot(residuals_tree, bins=50, kde=True, color="salmon", label="Decision Tree")
    plt.title("Residual Distribution Comparison", fontsize=13)
    plt.xlabel("Residuals")
    plt.legend()
    plt.show()


def plot_actual_vs_predicted(y_test, y_pred_lr, y_pred_tree):
    sample_df = pd.DataFrame({
        "Actual Sales": y_test.values.flatten(),
        "Predicted_LR": y_pred_lr.flatten(),
        "Predicted_Tree": y_pred_tree.flatten()
    }).sample(30, random_state=42)

    plt.figure(figsize=(12, 5))
    plt.plot(sample_df["Actual Sales"].values, label="Actual", color="black", linewidth=2)
    plt.plot(sample_df["Predicted_LR"].values, label="Linear Regression", linestyle="--", linewidth=2)
    plt.plot(sample_df["Predicted_Tree"].values, label="Decision Tree", linestyle="--", linewidth=2)
    plt.title("📉 Actual vs Predicted Sales (Sample 30 points)", fontsize=13)
    plt.xlabel("Sample Index")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()


# ============================================================
#  📌 MAIN PIPELINE
# ============================================================
if __name__ == "__main__":

    # Load splits
    X_train, X_test, y_train, y_test = load_data()

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Train models
    lr_model = train_linear_regression(X_train_scaled, y_train)
    tree_model = train_decision_tree(X_train, y_train)

    # Predictions
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_tree = tree_model.predict(X_test)

    # Evaluate
    results = []
    results.append(evaluate_model(y_test, y_pred_lr, "Linear Regression"))
    results.append(evaluate_model(y_test, y_pred_tree, "Decision Tree"))

    results_df = pd.DataFrame(results).set_index("Model")
    print("📈 Model Performance Comparison:")
    print(results_df.round(4))

    # Visualizations
    plot_model_performance(results_df)
    plot_feature_importance(tree_model, X_train)
    plot_residuals(y_test, y_pred_lr, y_pred_tree)
    plot_actual_vs_predicted(y_test, y_pred_lr, y_pred_tree)

    # Summary
    best_model = results_df["R²"].idxmax()
    print(f"🏆 Best Performing Model: {best_model}")
    print("=" * 70)
    print("📊 Final Model Metrics Summary:")
    print(results_df.round(4))
    print("🎯 Day 2 completed successfully - Baseline models built & compared professionally.")