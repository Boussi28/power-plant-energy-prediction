# Machine learning models to predict power plant energy output (PE)
# Models implemented:
# - Dummy baseline
# - Linear Regression
# - Random Forest with GridSearchCV hyperparameter tuning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Load dataset
data = pd.read_csv("data/power_plant_data.csv")

print("First five rows of the dataset:")
print(data.head())
print("\nColumns in dataset:")
print(data.columns)

# Features and target
X = data.drop("PE", axis=1)
y = data["PE"]

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

# Random Forest hyperparameter tuning
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
rf_pred = grid_search_rf.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)

# Print results
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

# Grid search results
cv_results = pd.DataFrame(grid_search_rf.cv_results_)

plt.figure(figsize=(8, 6))
for depth in param_grid["max_depth"]:
    temp = cv_results[cv_results["param_max_depth"] == depth]
    plt.plot(
        temp["param_n_estimators"],
        -temp["mean_test_score"],
        marker="o",
        label=f"max_depth={depth}"
    )

plt.xlabel("Number of Trees")
plt.ylabel("Mean Squared Error")
plt.title("Effect of n_estimators on Random Forest")
plt.legend()
plt.show()

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

lr_cv_scores = cross_val_score(lr, X, y, cv=kfold, scoring="r2")
rf_cv_scores = cross_val_score(grid_search_rf.best_estimator_, X, y, cv=kfold, scoring="r2")

print("\nLinear Regression CV Scores:", lr_cv_scores)
print("Average LR CV Score:", np.mean(lr_cv_scores))

print("\nRandom Forest CV Scores:", rf_cv_scores)
print("Average RF CV Score:", np.mean(rf_cv_scores))

# Plot cross-validation scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(lr_cv_scores) + 1), lr_cv_scores, marker="o", label="Linear Regression")
plt.plot(range(1, len(rf_cv_scores) + 1), rf_cv_scores, marker="o", label="Random Forest")

plt.xlabel("Fold")
plt.ylabel("R² Score")
plt.title("Cross-validation Scores")
plt.legend()
plt.show()

# Model comparison plots
models = ["Baseline", "Linear Regression", "Random Forest"]
r2_scores = [baseline_r2, lr_r2, rf_r2]
mse_scores = [baseline_mse, lr_mse, rf_mse]

plt.figure(figsize=(8, 5))
plt.bar(models, r2_scores)
plt.title("R² Comparison")
plt.ylabel("R²")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(models, mse_scores)
plt.title("MSE Comparison")
plt.ylabel("MSE")
plt.show()
