# Power Plant Energy Prediction

## Overview

This project develops and evaluates machine learning models to predict the **net hourly electrical energy output (PE)** of a combined cycle power plant using environmental measurements.

The task is formulated as a **supervised regression problem**, where the goal is to learn a function that maps environmental conditions to power output.

Three models are implemented and compared:

- Baseline model (Dummy Regressor)
- Linear Regression
- Random Forest Regression with hyperparameter tuning

The models are evaluated using **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R² score**, and their robustness is assessed using **k-fold cross-validation**.

---

# Dataset

The dataset contains operational measurements from a **Combined Cycle Power Plant** collected over several years.

The objective is to predict the **net hourly electrical energy output (PE)** from environmental conditions.

### Input Features

| Feature | Description |
|------|------|
| AT | Ambient Temperature |
| V | Exhaust Vacuum |
| AP | Ambient Pressure |
| RH | Relative Humidity |

### Target Variable

| Variable | Description |
|------|------|
| PE | Net hourly electrical energy output |

The dataset is stored in the repository:

```
data/power_plant_data.csv
```

The dataset contains **9568 observations and 5 variables**, with no missing values.

---

# Problem Formulation

This is a **supervised regression task**.

The goal is to learn a function:

```
f : X → ℝ
```

where the feature space is:

```
X = {AT, V, AP, RH}
```

and the output variable is:

```
PE
```

The objective is to minimise the prediction error between predicted and actual power output.

---

# Exploratory Data Analysis

Before training the models, basic exploratory analysis is performed:

- dataset shape and summary statistics  
- missing value inspection  
- correlation analysis between variables  

A **correlation heatmap** is generated to visualise relationships between features.

Key observation:

- **Ambient Temperature (AT)** shows the strongest negative correlation with power output.

---

# Machine Learning Models

## Baseline Model

A **Dummy Regressor** is used as a baseline model.

This model simply predicts the **mean value of the target variable (PE)** for all observations.

The baseline provides a reference to determine whether machine learning models learn meaningful patterns from the data.

---

## Linear Regression

Linear regression models the relationship between input variables and the target variable using a linear equation.

The model estimates coefficients that minimise the **residual sum of squares (RSS)** between predicted and actual values.

Advantages:

- simple and interpretable  
- provides insight into how each feature affects power output  

---

## Random Forest Regression

A **Random Forest Regressor** is used to capture more complex relationships in the data.

Random forests combine many decision trees to produce a robust ensemble prediction.

Advantages:

- captures nonlinear relationships  
- handles feature interactions  
- reduces overfitting through ensemble averaging  

---

# Hyperparameter Tuning

The Random Forest model is optimised using **GridSearchCV**.

The hyperparameters explored were:

```
n_estimators = [100, 200, 300]
max_depth = [None, 5, 10]
```

Where:

- **n_estimators** controls the number of trees in the forest  
- **max_depth** controls the maximum depth of each tree  

GridSearchCV evaluates all parameter combinations using cross-validation and selects the model that minimises **negative mean squared error**.

The best hyperparameters found were:

```
max_depth = None
n_estimators = 300
```

---

# Model Training

The dataset is split into:

```
Training data: 80%
Testing data: 20%
```

Models are trained on the training set and evaluated on the test set to assess predictive performance.

---

# Evaluation Metrics

Three metrics are used to evaluate the models.

### Mean Squared Error (MSE)

MSE measures the average squared difference between predicted and actual values.

Lower values indicate better model performance.

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE and provides an error metric in the same units as the target variable.

### R² Score

The R² score measures the proportion of variance in the target variable explained by the model.

Higher values indicate better predictive performance.

---

# Results

| Model | MSE | RMSE | R² |
|------|------|------|------|
| Baseline | ~290 | ~17.03 | ~0 |
| Linear Regression | ~20.27 | ~4.50 | ~0.93 |
| Random Forest | ~10.43 | ~3.23 | ~0.96 |

### Interpretation

The baseline model performs poorly because it does not use any input information.

Linear regression performs very well, explaining approximately **93% of the variance in power output**.

The Random Forest model achieves the best performance, reducing the prediction error further and achieving an **R² score of approximately 0.96**.

This suggests that **nonlinear relationships and feature interactions** exist in the dataset that are better captured by the Random Forest model.

---

# Feature Importance

Random Forest also provides an estimate of **feature importance**.

Results show that:

- **Ambient Temperature (AT)** is the most influential predictor of power output.  
- Other variables (**V, AP, RH**) contribute smaller but useful predictive information.

---

# Cross Validation

To evaluate model robustness, **5-fold cross-validation** is used.

Average cross-validation scores:

| Model | Average R² |
|------|------|
| Linear Regression | ~0.93 |
| Random Forest | ~0.96 |

These results confirm that both models generalise well to unseen data.

---

# Visualisations

The script automatically generates and saves several plots in the `results/` folder:

- Correlation heatmap  
- Feature importance plot  
- Hyperparameter tuning results  
- Cross-validation scores  
- Model comparison plots (R², MSE, RMSE)  
- Actual vs predicted values  
- Residual analysis  

These visualisations help interpret model performance and diagnostics.

---

# Repository Structure

```
power-plant-energy-prediction
│
├── data
│   └── power_plant_data.csv
│
├── src
│   └── power_plant_model.py
│
├── results
│   └── generated plots
│
└── README.md
```

---

# Libraries Used

- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

# Running the Project

Install the required libraries:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

Run the model script:

```
python src/power_plant_model.py
```

The script will:

1. Load and analyse the dataset  
2. Train machine learning models  
3. Evaluate model performance  
4. Perform cross-validation  
5. Generate and save visualisations  

All plots will be saved in the **results/** folder.

---

# Conclusion

Both machine learning models significantly outperform the baseline model.

While linear regression provides a strong and interpretable baseline, the **Random Forest model achieves the best predictive performance**, indicating that nonlinear patterns and feature interactions exist in the dataset.

Future work could explore:

- additional ensemble methods  
- deeper hyperparameter tuning  
- neural network models for regression  

---

# Author

**Bouthaina Hachemi**
