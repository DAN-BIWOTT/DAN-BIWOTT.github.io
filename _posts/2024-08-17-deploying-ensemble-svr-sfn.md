---
title: "Deploying an Ensembled Forecasting Model (Linear SVR + SFN)"
date: 2025-06-17
tags: [Machine Learning, Forecasting, Ensemble, ONNX, Python]
categories: [Projects]
description: "A step-by-step walkthrough of building, training, evaluating, and exporting an ensemble model using Linear SVR and SFN for market prediction."
--- 

This post takes you under the hood of a custom-built forecasting engine—a hybrid model that blends the precision of a **Linear Support Vector Regressor (SVR)** with the pattern-recognition strength of a **Simple Feedforward Network (SFN)**, both implemented in scikit-learn.  

Think of it like pairing a tactical marksman with a perceptive analyst: the SVR locks onto linear patterns with precision, while the SFN explores nonlinear terrains often invisible to traditional models.  
Together, they team up to forecast the next median candlestick price in the USD/JPY forex market. The inputs? A carefully engineered set of signals drawn from thousands of lines of historical trading data—each feature acting like a sensor feeding real-time battlefield intelligence into the model’s decision-making core.

 

## 🔹 Step 1: Import Dependencies

Before we can fire up our forecasting engine, we need to gather our toolkit. These libraries function like a command squad:

- `pandas` for data logistics  
- `tensorflow` *(optional, for future deep learning extensions)*  
- `matplotlib.pyplot` for visual diagnostics  
- `re` and `sys` for command-line interface parsing and automation  

```python
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import sys
````

 

## 🔹 Step 2: Load and Preprocess Market Data

1. Accept a CSV file via command line
2. Extract date for versioning
3. Reverse data chronologically (newest last)
4. Limit to the most recent 10,000 rows
5. Engineer features such as:

   * **Future_Median** (shifted midpoint of next candlestick)
   * **Price_Difference** between highs and lows
   * **Percentage change** between open and close
   * **Rolling volume mean** (20-period moving average)
   * **Lagged medians** `median_t-1`, `median_t-2` for short-term temporal context

Each feature helps the model “see” the market from multiple tactical vantage points.

 

## 🔹 Step 3: Feature Selection

We define our feature matrix `X` and target `y`:

```python
features = [
    "Open", "High", "Low", "Close", "Price_Difference",
    "Open_Close_Change_Pct", "High_Low_Change_Pct",
    "Volume", "Volume_MA_20", "Volume_Change_Pct",
    "median_t-1", "median_t-2"
]
```

All rows with missing values are dropped to maintain structural integrity in the time series.

 

## 🔹 Step 4: Scaling and Splitting

Once features are engineered, data is split into training and testing sets and standardized to ensure stable learning:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Scaling ensures that no single feature dominates the training due to unit magnitude differences—critical when combining linear and nonlinear learners.

 

## 🔹 Step 5: Train SVR and SFN with Grid Search

Both models undergo hyperparameter tuning via `GridSearchCV`.

### **Linear SVR Parameters**

```python
param_grid = {'C': [1, 10, 50], 'epsilon': [0.01, 0.1, 1]}
```

### **SFN Parameters (MLPRegressor)**

```python
param_grid_sfn = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}
```

The grid search systematically tests multiple model configurations, optimizing for **lowest mean squared error (MSE)**.
Once complete, each model is retrieved with its best-performing hyperparameters.

 

## 🔹 Step 6: Ensemble the Models

Now we let both specialists collaborate—using `VotingRegressor` to combine their perspectives.
LinearSVR provides structure; SFN supplies nuance.

```python
from sklearn.ensemble import VotingRegressor

ensemble_model = VotingRegressor(
    estimators=[
        ('linear_svr', best_linear_svr_model),
        ('sfn', best_sfn_model)
    ],
    weights=[0.7, 0.3]  # LinearSVR dominates due to its stability
)
ensemble_model.fit(X_train, y_train)
```

This approach balances speed, interpretability, and adaptability.

 

## 🔹 Step 7: Evaluate Model Performance

We compute multiple diagnostic metrics:

* **MSE (Mean Squared Error)**
* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**
* **R² (Coefficient of Determination)**
* **Explained Variance**

Plots visually compare actual vs. predicted median prices across both training and testing sets:

```python
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted', color='red', linestyle='--')
plt.title('Testing Data: Actual vs. Predicted Median Price')
plt.xlabel('Index')
plt.ylabel('Median Price')
plt.legend()
plt.show()
```

 

## 🔹 Step 8: Save Logs and Export to ONNX

The ensemble and its environment data are serialized for deployment.
Performance metrics, scaling stats, and feature lists are logged in JSON.

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('input', FloatTensorType([None, X_test.shape[1]]))]
onnx_model = convert_sklearn(ensemble_model, initial_types=initial_type)
```

Saving in **ONNX** format allows integration beyond Python—compatible with trading terminals and embedded inference systems like MetaTrader 5.

 

## 🔹 Step 9: Send Model Log to Arasaka Neural Bastion (Optional API Endpoint)

When the EC2 instance is online, metadata can be pushed to a central monitoring endpoint for audit and tracking.

```python
result = SendToArasaka(log_data)
print(result)
```

This ensures every model version, scaling factor, and performance metric is archived in the **Arasaka Neural Bastion**—a form of mission log for your forecasting AI.

 

## ⚙️ Deployment Context

* **Platform:** AWS EC2 (Windows environment)
* **Automation:** `run_latest.bat` triggers data extraction, training, and ONNX deployment
* **Integration:**

  * `PrinceSwingEnsembled.mq5` → Demo trading
  * `AutoTuesdayProdPE.mq5` → Live trading
* **Target Terminals:** Dual MetaTrader installations for safe and live execution

 

## 🧩 Summary

This ensemble marks a tactical evolution in quantitative forecasting—balancing interpretability with adaptive intelligence.
The **Linear SVR** offers the calm discipline of a marksman, while the **SFN** reacts with creative intuition to nonlinear shifts in the market battlefield.

By exporting to ONNX, this model can operate natively within non-Python ecosystems, setting the foundation for automated live trading agents.

**Stay tuned for Part 2:**
*Advanced Feature Engineering + Hyperparameter Tuning: From Market Volatility to Momentum Signals.*
