# 🔋 Energy Consumption Prediction
### A Machine Learning Approach to Building Energy Forecasting


---

## 📋 Project Overview

This project predicts hourly building energy consumption using machine learning. Three models were implemented and compared — **Linear Regression**, **Random Forest**, and **XGBoost** — using a dataset of 1,000 hourly observations across environmental, structural, and operational building features.

**Best Model:** XGBoost — Test R² = 0.54, MAE = 4.41, RMSE = 5.46

---

## 📁 Repository Structure

```
├── main.ipynb                  # Main Jupyter Notebook (full pipeline)
├── Energy_consumption.csv      # Dataset
├── README.md                   # This file
```

---

## 📦 Requirements

Make sure you have **Python 3.10+** installed. Then install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels xgboost
```

Or install individually:

```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install statsmodels
pip install xgboost
```

---

## 📂 How to Import the Data

Place `Energy_consumption.csv` in the same directory as the notebook, then load it:

```python
import pandas as pd

df = pd.read_csv('Energy_consumption.csv')
df.head()
```

The dataset contains **1,000 rows** and **11 columns:**

| Column | Type | Description |
|---|---|---|
| Timestamp | Datetime | Hourly timestamp |
| Temperature | Float | Outdoor temperature (°C) |
| Humidity | Float | Relative humidity (%) |
| SquareFootage | Float | Building floor area (sq ft) |
| Occupancy | Integer | Number of occupants |
| HVACUsage | String | HVAC system status (On/Off) |
| LightingUsage | String | Lighting system status (On/Off) |
| RenewableEnergy | Float | Renewable energy generated |
| DayOfWeek | String | Day of the week |
| Holiday | String | Public holiday (Yes/No) |
| EnergyConsumption | Float | **Target variable** |

---

## ⚙️ How to Run the Model

### Step 1 — Feature Engineering

```python
import numpy as np

# Extract hour from timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hours'] = df['Timestamp'].dt.hour

# Cyclic encoding for hour of day
df['Hour_sin'] = np.sin(2 * np.pi * df['Hours'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hours'] / 24)

# Encode categorical binary features
df['HVACUsage']     = df['HVACUsage'].map({'On': 1, 'Off': 0})
df['LightingUsage'] = df['LightingUsage'].map({'On': 1, 'Off': 0})
```

### Step 2 — Define Features and Target

```python
X = df[['Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
        'RenewableEnergy', 'HVACUsage', 'LightingUsage', 'Hour_cos']]
Y = df['EnergyConsumption']
```

### Step 3 — Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)
```

### Step 4 — Run Linear Regression

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

cv_scores = cross_val_score(pipeline_lr, X_train, Y_train, cv=5, scoring='r2')
print(f"CV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f}")

pipeline_lr.fit(X_train, Y_train)
Y_pred_lr = pipeline_lr.predict(X_test)
```

### Step 5 — Run Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline_rf.fit(X_train, Y_train)
Y_pred_rf = pipeline_rf.predict(X_test)
```

### Step 6 — Run XGBoost (Best Model)

```python
from xgboost import XGBRegressor

pipeline_xgb = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

pipeline_xgb.fit(X_train, Y_train)
Y_pred_xgb = pipeline_xgb.predict(X_test)
```

### Step 7 — Evaluate All Models

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import root_mean_squared_error

models = {
    'Linear Regression': Y_pred_lr,
    'Random Forest':     Y_pred_rf,
    'XGBoost':           Y_pred_xgb
}

for name, preds in models.items():
    r2   = r2_score(Y_test, preds)
    mae  = mean_absolute_error(Y_test, preds)
    rmse = root_mean_squared_error(Y_test, preds)
    mse  = mean_squared_error(Y_test, preds)
    print(f"{name}: R²={r2:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f} | MSE={mse:.4f}")
```

---

## 📊 Results

| Model | CV R² | Test R² | MAE | RMSE | MSE
|---|---|---|---|---|
| Linear Regression | 0.62 | 0.33 | 4.10 | 5.12 | 26.28
| Random Forest | 0.54 | 0.25 | 4.35 | 5.45 | 29.78
| **XGBoost** | **0.55** | **0.54** | **4.42** | **5.48** | 30.06

> **Note:** The dataset is synthetically generated, evidenced by uniform feature distributions and the absence of natural temporal patterns. The theoretical R² ceiling without explicit time-series modelling is approximately 0.6.

---

## 🔍 VIF Analysis (Multicollinearity Check)

```python
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = add_constant(df[['Temperature', 'Humidity', 'SquareFootage', 'RenewableEnergy']])

for i in range(X_vif.shape[1]):
    vif = variance_inflation_factor(X_vif.values, i)
    print(f"{X_vif.columns[i]}: VIF = {vif:.3f}")
```

> ⚠️ **Important:** Always use `add_constant()` before computing VIF. Omitting it produces artificially inflated values and incorrect conclusions.

---

## 📈 Plots

```python
import matplotlib.pyplot as plt

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred_xgb, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()],
         [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('XGBoost — Actual vs Predicted')
plt.show()

# Residual Plot
residuals = Y_test - Y_pred_xgb
plt.figure(figsize=(8, 6))
plt.scatter(Y_pred_xgb, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('XGBoost — Residual Plot')
plt.show()
```

---

## 🛠️ Environment

- Python 3.10+
- Jupyter Notebook / Google Colab
- All libraries listed in Requirements section above

---

## 📄 References

1. Mosavi, A., & Bahmani, A. (2019). Energy consumption prediction using machine learning; a review. Preprints. https://doi.org/10.20944/preprints201903.0131.v1.

2.Pham, A.-D., Ngo, N.-T., Truong, T. T. H., Huynh, N.-T., & Truong, N.-S. (2020). Predicting energy consumption in multiple buildings using machine learning for improving energy efficiency and sustainability. Journal of Cleaner Production, 260, 121082. https://doi.org/10.1016/j.jclepro.2020.121082.

3.García-Martín, E., Rodrigues, C. F., Riley, G., & Grahn, H. (2019). Estimation of energy consumption in machine learning. Journal of Parallel and Distributed Computing, 134, 75–88. https://doi.org/10.1016/j.jpdc.2019.07.007.

4.Mhlanga, D. (2023). Artificial intelligence and machine learning for energy consumption and production in emerging markets: A review. Energies, 16(2), 745. https://doi.org/10.3390/en16020745
