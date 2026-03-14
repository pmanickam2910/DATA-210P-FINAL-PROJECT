# 🔋 Building Energy Consumption Prediction
### A Machine Learning Approach to Building Energy Forecasting

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![XGBoost](https://img.shields.io/badge/xgboost-latest-red)

---

## 📋 Project Overview

This project predicts hourly building energy consumption using machine learning. Three models were implemented and compared — **Linear Regression**, **Random Forest**, and **XGBoost** — using a dataset of 1,000 hourly observations across environmental, structural, and operational building features.

**Best Performing Model:** XGBoost (Test R² = 0.54, MAE = 4.41, RMSE = 5.46)

---

## 📁 Repository Structure

```text
├── main.ipynb                  # Main Jupyter Notebook (full pipeline)
├── Energy_consumption.csv      # Dataset
└── README.md                   # Project documentation
```

---

## 📦 Requirements

This project requires **Python 3.10+**. You can install the required dependencies using `pip`:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels xgboost
```

---

## 📂 Dataset

Place `Energy_consumption.csv` in the same directory as the notebook. The dataset contains **1,000 rows** and **11 columns**:

| Column | Type | Description |
|---|---|---|
| `Timestamp` | Datetime | Hourly timestamp |
| `Temperature` | Float | Outdoor temperature (°C) |
| `Humidity` | Float | Relative humidity (%) |
| `SquareFootage` | Float | Building floor area (sq ft) |
| `Occupancy` | Integer | Number of occupants |
| `HVACUsage` | String | HVAC system status (On/Off) |
| `LightingUsage` | String | Lighting system status (On/Off) |
| `RenewableEnergy` | Float | Renewable energy generated |
| `DayOfWeek` | String | Day of the week |
| `Holiday` | String | Public holiday (Yes/No) |
| `EnergyConsumption` | Float | **Target variable** |

---

## ⚙️ Usage & Modeling Pipeline

Below is the step-by-step implementation of the data processing and modeling pipeline.

### 1. Feature Engineering & Preprocessing
Extracts cyclical time features and encodes categorical variables:

```python
import numpy as np
import pandas as pd

df = pd.read_csv('Energy_consumption.csv')

# Extract hour and apply cyclic encoding
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hours'] = df['Timestamp'].dt.hour
df['Hour_sin'] = np.sin(2 * np.pi * df['Hours'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hours'] / 24)

# Encode categorical binary features
df['HVACUsage'] = df['HVACUsage'].map({'On': 1, 'Off': 0})
df['LightingUsage'] = df['LightingUsage'].map({'On': 1, 'Off': 0})

# Define Features and Target
X = df[['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 
        'RenewableEnergy', 'HVACUsage', 'LightingUsage', 'Hour_cos']]
Y = df['EnergyConsumption']
```

### 2. Model Training
Splits the data and trains the three distinct models using Scikit-Learn pipelines:

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Linear Regression
pipeline_lr = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
pipeline_lr.fit(X_train, Y_train)
Y_pred_lr = pipeline_lr.predict(X_test)

# Random Forest
pipeline_rf = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
pipeline_rf.fit(X_train, Y_train)
Y_pred_rf = pipeline_rf.predict(X_test)

# XGBoost (Best Model)
pipeline_xgb = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, 
                           subsample=0.8, colsample_bytree=0.8, random_state=42))
])
pipeline_xgb.fit(X_train, Y_train)
Y_pred_xgb = pipeline_xgb.predict(X_test)
```

---

## 📊 Results

| Model | CV R² | Test R² | MAE | RMSE | MSE |
|---|---|---|---|---|---|
| Linear Regression | 0.62 | 0.33 | 4.10 | 5.12 | 26.28 |
| Random Forest | 0.54 | 0.25 | 4.35 | 5.45 | 29.78 |
| **XGBoost** | **0.55** | **0.54** | **4.42** | **5.48** | **30.06** |

> **Note:** The dataset used is synthetically generated, evidenced by uniform feature distributions and the absence of natural temporal patterns. The theoretical R² ceiling without explicit time-series modeling is approximately 0.6.

---

## 🔍 Multicollinearity Check (VIF)

To ensure feature independence, Variance Inflation Factor (VIF) analysis was conducted:

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

## 📈 Visualizations

The script generates visual diagnostics for the best performing model (XGBoost):

```python
import matplotlib.pyplot as plt

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred_xgb, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel('Actual Energy Consumption')
plt.ylabel('Predicted Energy Consumption')
plt.title('XGBoost — Actual vs Predicted')
plt.show()

# Residual Plot
residuals = Y_test - Y_pred_xgb
plt.figure(figsize=(8, 6))
plt.scatter(Y_pred_xgb, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Energy Consumption')
plt.ylabel('Residuals')
plt.title('XGBoost — Residual Plot')
plt.show()
```

---

## 📄 References

1. Mosavi, A., & Bahmani, A. (2019). Energy consumption prediction using machine learning; a review. *Preprints*. [https://doi.org/10.20944/preprints201903.0131.v1](https://doi.org/10.20944/preprints201903.0131.v1)
2. Pham, A.-D., Ngo, N.-T., Truong, T. T. H., Huynh, N.-T., & Truong, N.-S. (2020). Predicting energy consumption in multiple buildings using machine learning for improving energy efficiency and sustainability. *Journal of Cleaner Production*, 260, 121082. [https://doi.org/10.1016/j.jclepro.2020.121082](https://doi.org/10.1016/j.jclepro.2020.121082)
3. García-Martín, E., Rodrigues, C. F., Riley, G., & Grahn, H. (2019). Estimation of energy consumption in machine learning. *Journal of Parallel and Distributed Computing*, 134, 75–88. [https://doi.org/10.1016/j.jpdc.2019.07.007](https://doi.org/10.1016/j.jpdc.2019.07.007)
4. Mhlanga, D. (2023). Artificial intelligence and machine learning for energy consumption and production in emerging markets: A review. *Energies*, 16(2), 745. [https://doi.org/10.3390/en16020745](https://doi.org/10.3390/en16020745)
