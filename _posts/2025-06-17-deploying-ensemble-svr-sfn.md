---
title: "Deploying an Ensembled Forecasting Model (Linear SVR + SFN)"
date: 2025-06-17
tags: [Machine Learning, Forecasting, Ensemble, ONNX, Python]
categories: [Projects]
description: "A step-by-step walkthrough of building, training, evaluating, and exporting an ensemble model using Linear SVR and SFN for market prediction."
---

This post takes you under the hood of a custom-built forecasting engine—a hybrid model that blends the precision of a **Linear Support Vector Regressor (SVR)** with the pattern-recognition strength of a **Simple Feedforward Network (SFN)**, both implemented in scikit-learn. Think of it like pairing a tactical marksman with a perceptive analyst: the SVR locks onto linear patterns with precision, while the SFN explores nonlinear terrains often invisible to traditional models. Together, they forecast the ne...

---

**🔹 Step 1: Import Dependencies**

Before we can fire up our forecasting engine, we need to gather our toolkit. These libraries function like a command squad:

- `pandas` for data logistics
- `tensorflow` (optional, for future deep learning extensions)
- `matplotlib.pyplot` for visual diagnostics
- `re` and `sys` for command-line interface parsing and automation

```python
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import sys
```

---

**🔹 Step 2: Load and Preprocess Market Data**

- Accept a CSV file via command line
- Extract date for versioning
- Reverse data chronologically
- Limit to most recent 10,000 rows
- Engineer features including `Future_Median`, percent changes, volume averages, and lagged medians

---

**🔹 Step 3: Feature Selection**

We define our feature matrix `X` and target `y`:

```python
features = ["Open", "High", "Low", "Close", "Price_Difference", "Open_Close_Change_Pct",
            "High_Low_Change_Pct", "Volume", "Volume_MA_20", "Volume_Change_Pct",
            "median_t-1", "median_t-2"]
```

Missing values are dropped to maintain integrity.

---

**🔹 Step 4: Scaling and Splitting**

Data is scaled and partitioned into training and testing sets:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

**🔹 Step 5: Train SVR and SFN with Grid Search**

**LinearSVR Parameters:**
```python
param_grid = {'C': [1, 10, 50], 'epsilon': [0.01, 0.1, 1]}
```

**SFN Parameters (MLPRegressor):**
```python
param_grid_sfn = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}
```

---

**🔹 Step 6: Ensemble the Models**

Using `VotingRegressor` to blend predictions:

```python
from sklearn.ensemble import VotingRegressor
ensemble_model = VotingRegressor(estimators=[('linear_svr', best_linear_svr_model),
                                             ('sfn', best_sfn_model)],
                                 weights=[0.7, 0.3])
ensemble_model.fit(X_train, y_train)
```

---

**🔹 Step 7: Evaluate Model Performance**

We compute the following metrics:

- MSE, RMSE, MAE
- R² (Coefficient of Determination)
- Explained Variance

Visual plots compare actual vs. predicted medians for both training and testing sets.

---

**🔹 Step 8: Save Logs and Export to ONNX**

- Save performance logs to JSON
- Include model version, feature list, and scaling parameters

Export model to ONNX:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
onnx_model = convert_sklearn(ensemble_model, initial_types=[('input', FloatTensorType([None, X_test.shape[1]]))])
```

---

**🔹 Step 9: Send Model Log to Arasaka Neural Bastion (Optional API Endpoint)**

If online, performance metadata can be sent to a remote API for archival and monitoring:

```python
result = SendToArasaka(log_data)
print(result)
```

---

**Summary**

This ensemble marks a step forward in tactical forecasting—balancing interpretability with flexibility. Exporting to ONNX means the model can now serve in non-Python environments (like trading terminals or embedded agents). Stay tuned for Part 2: **Advanced Feature Engineering + Hyperparameter Tuning**.
