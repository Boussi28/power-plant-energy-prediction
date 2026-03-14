"""
Power Plant Energy Prediction
Machine learning regression models predicting net hourly energy output (PE)
from environmental variables.

Models implemented:
- Dummy baseline
- Linear Regression
- Random Forest with hyperparameter tuning

Evaluation metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² score
- Cross-validation
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility
np.random.seed(42)

# Output folder for saved plots
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
data = pd.read_csv("data/power_plant_data.csv")

print("First five rows of the dataset:")
print(data.head())

print("\nColumns in dataset:")
print(data.columns)

print("\nDataset shape:")
print(data.shape)

print("\nMissing values:")
print(data.isnull().sum())

print("\nSummary statistics:")
print(data.describe())

print("\nCorrelation Matrix:")
print(data.corr())

# Features and target
X = data.drop("PE", axis=1)
y = data["PE"]

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
plt.close()

print("\nSaved: correlation_matrix.png")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline model
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)
dummy_pred = dummy_regr.predict(X_test)

baseline_r2 = r2_score(y_test, dummy_pred)
baseline_mse = mean_squared_error(y_test, dummy_pred)
baseline_rmse = np.sqrt(baseline_mse)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_r2 = r2_score(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)

print("\nLinear Regression Coefficients:")
for feature, coef in zip(X.columns, lr.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {lr.intercept_:.4f}")

# Random Forest with GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10]
}

rf = RandomForestRegressor(random_state=42)

grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
rf_pred = best_rf.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)

# Feature importance
feature_importance = best_rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (Random Forest):")
for _, row in importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.3f}")

plt.figure(figsize=(8, 5))
plt.bar(importance_df["Feature"], importance_df["Importance"])
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()

print("Saved: feature_importance.png")

# Results
print("\nBest Hyperparameters:", grid_search_rf.best_params_)

print("\nBaseline Model")
print("R²:", baseline_r2)
print("MSE:", baseline_mse)
print("RMSE:", baseline_rmse)

print("\nLinear Regression")
print("R²:", lr_r2)
print("MSE:", lr_mse)
print("RMSE:", lr_rmse)

print("\nRandom Forest")
print("R²:", rf_r2)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)

# Grid search results plot
cv_results = pd.DataFrame(grid_search_rf.cv_results_)

plt.figure(figsize=(8, 6))
for depth in param_grid["max_depth"]:
    temp = cv_results[cv_results["param_max_depth"] == depth]
    plt.plot(
        temp["param_n_estimators"].astype(int),
        -temp["mean_test_score"],
        marker="o",
        label=f"max_depth={depth}"
    )

plt.xlabel("Number of Trees")
plt.ylabel("Mean Squared Error")
plt.title("Effect of n_estimators on Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rf_hyperparameter_effect.png"))
plt.close()

print("Saved: rf_hyperparameter_effect.png")

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

lr_cv_scores = cross_val_score(lr, X, y, cv=kfold, scoring="r2")
rf_cv_scores = cross_val_score(best_rf, X, y, cv=kfold, scoring="r2")

print("\nLinear Regression CV Scores:", lr_cv_scores)
print("Average LR CV Score:", np.mean(lr_cv_scores))

print("\nRandom Forest CV Scores:", rf_cv_scores)
print("Average RF CV Score:", np.mean(rf_cv_scores))

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(lr_cv_scores) + 1), lr_cv_scores, marker="o", label="Linear Regression")
plt.plot(range(1, len(rf_cv_scores) + 1), rf_cv_scores, marker="o", label="Random Forest")
plt.xlabel("Fold")
plt.ylabel("R² Score")
plt.title("Cross-validation Scores")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cross_validation_scores.png"))
plt.close()

print("Saved: cross_validation_scores.png")

# Model comparison plots
models = ["Baseline", "Linear Regression", "Random Forest"]
r2_scores = [baseline_r2, lr_r2, rf_r2]
mse_scores = [baseline_mse, lr_mse, rf_mse]
rmse_scores = [baseline_rmse, lr_rmse, rf_rmse]

plt.figure(figsize=(8, 5))
plt.bar(models, r2_scores)
plt.title("R² Comparison")
plt.ylabel("R²")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "r2_comparison.png"))
plt.close()

print("Saved: r2_comparison.png")

plt.figure(figsize=(8, 5))
plt.bar(models, mse_scores)
plt.title("MSE Comparison")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mse_comparison.png"))
plt.close()

print("Saved: mse_comparison.png")

plt.figure(figsize=(8, 5))
plt.bar(models, rmse_scores)
plt.title("RMSE Comparison")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rmse_comparison.png"))
plt.close()

print("Saved: rmse_comparison.png")

# Actual vs Predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.xlabel("Actual Power Output")
plt.ylabel("Predicted Power Output")
plt.title("Actual vs Predicted (Random Forest)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "actual_vs_predicted_rf.png"))
plt.close()

print("Saved: actual_vs_predicted_rf.png")

# Residual plot
residuals = y_test - rf_pred

plt.figure(figsize=(6, 5))
plt.scatter(rf_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residual_plot_rf.png"))
plt.close()

print("Saved: residual_plot_rf.png")

print("\nModel training complete.")
print("Random Forest provided the best performance based on MSE and R².")
print(f"\nAll plots were saved to: {output_dir}")
