# Power Plant Energy Prediction

## Overview

This project develops and evaluates machine learning models to predict the **net hourly electrical energy output (PE)** of a combined cycle power plant using environmental measurements.

The task is formulated as a **supervised regression problem**, where the goal is to learn a function that maps environmental conditions to power output.

Three models are implemented and compared:

- Baseline model (Dummy Regressor)
- Linear Regression
- Random Forest Regression

The models are evaluated using **Mean Squared Error (MSE)** and **R² score**, and their robustness is assessed using **k-fold cross-validation**.

---

# Dataset

The dataset contains operational measurements from a **Combined Cycle Power Plant** collected over several years.

The goal is to predict the net hourly energy output of the plant.

### Input Features

| Feature | Description |
|------|------|
| AT | Temperature |
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

---

# Problem Formulation

The task is a **regression problem**, where the objective is to learn a function:

```
f : V → ℝ
```

where the feature space is:

```
V = {AT, V, AP, RH}
```

and the output variable is:

```
PE
```

The goal is to minimise the prediction error between predicted and actual energy output.

---

# Machine Learning Models

## Baseline Model

A **Dummy Regressor** is used as a baseline model.  
This model simply predicts the **mean value of the target variable (PE)** for every observation.

The baseline provides a reference point to determine whether machine learning models are actually learning meaningful relationships from the data.

---

## Linear Regression

Linear regression assumes a **linear relationship between input variables and the target variable**.

The model estimates coefficients that minimise the **residual sum of squares (RSS)** between predicted and actual values.

Linear regression is useful because:

- it is simple and interpretable
- it provides insight into the relationship between environmental variables and energy output

---

## Random Forest Regression

A **Random Forest Regressor** is used to capture more complex patterns in the data.

Random forests combine many decision trees to produce a more robust prediction.

Advantages of random forests include:

- ability to model nonlinear relationships
- capturing interactions between features
- reduced risk of overfitting through ensemble averaging

---

# Hyperparameter Tuning

The Random Forest model was optimised using **GridSearchCV**.

The hyperparameters explored were:

```
n_estimators = [100, 200, 300]
max_depth = [None, 5, 10]
```

Where:

- **n_estimators** controls the number of trees in the forest
- **max_depth** controls the maximum depth of each tree

GridSearchCV evaluates all combinations of these parameters using cross-validation and selects the model that minimises the **negative mean squared error**.

The best hyperparameters found were:

```
max_depth = None
n_estimators = 300
```

---

# Model Training

The dataset was split into:

```
Training data: 80%
Testing data: 20%
```

The models are trained on the training set and evaluated on the test set to measure their ability to predict unseen data.

---

# Evaluation Metrics

Two metrics were used to evaluate the models.

### Mean Squared Error (MSE)

MSE measures the average squared difference between predicted and actual values.

Lower MSE values indicate more accurate predictions.

### R² Score

The R² score measures the proportion of variance in the target variable explained by the model.

Higher R² values indicate better model performance.

---

# Results

| Model | MSE | R² |
|------|------|------|
| Baseline (Dummy) | ~290 | ~0 |
| Linear Regression | ~20 | ~0.93 |
| Random Forest | ~10 | ~0.96 |

### Interpretation of Results

The baseline model performs poorly, with a very high MSE and R² close to zero. This is expected because the model simply predicts the average energy output and does not use any input information.

Linear regression significantly improves performance, achieving an R² score of approximately **0.93**, indicating that about **93% of the variation in power output can be explained by the input variables**.

The Random Forest model achieves the best performance, with an R² score of approximately **0.96** and the lowest MSE.

This improvement suggests that the relationship between environmental variables and energy output may contain **nonlinear patterns or interactions** that are better captured by the Random Forest model.

---

# Cross Validation

To evaluate model robustness, **k-fold cross-validation** was performed.

```
k = 3
```

Average cross-validation scores:

| Model | Average R² |
|------|------|
| Linear Regression | ~0.93 |
| Random Forest | ~0.96 |

These results confirm that both models generalise well to unseen data.

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
└── README.md
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

# Running the Project

Install the required libraries:

```
pip install pandas numpy scikit-learn matplotlib
```

Run the model script:

```
python src/power_plant_model.py
```

The script will train the models, evaluate performance, and generate plots comparing the models.

---

# Conclusion

Both machine learning models significantly outperform the baseline model.

While linear regression provides a strong and interpretable baseline, the **Random Forest model achieves the best predictive performance**, indicating that the relationship between environmental conditions and power output may include nonlinear interactions.

Future work could explore additional models, deeper hyperparameter tuning, or neural network approaches to further improve predictive accuracy.

---

# Author

Machine learning coursework project exploring regression modelling and model evaluation for power plant energy prediction.
