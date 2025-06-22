---
title: "Deploying an Ensembled Forecasting Model (Linear SVR + SFN)"
date: 2025-06-17
tags: [Machine Learning, Forecasting, Ensemble, ONNX, Python]
categories: [Projects]
description: "A step-by-step walkthrough of building, training, evaluating, and exporting an ensemble model using Linear SVR and SFN for market prediction."
---

In this post, I take you under the hood of a custom-built forecasting engine—a hybrid model that fuses the precision of a Linear Support Vector Regressor (SVR) with the intuition of a Simple Feedforward Network (SFN), both crafted using scikit-learn. Think of it like pairing a sniper and a scout: one is linear and sharp with low variance; the other, nonlinear and perceptive, learning from patterns the first might miss. Together, they team up to forecast the next median candlestick price in the USD/JPY forex market. The inputs? A carefully engineered set of signals drawn from thousands of lines of historical trading data—each feature acting like a sensor feeding real-time battlefield intelligence into the model’s decision-making core.


---

🔹 Step 1: Import Dependencies
Before we can fire up our forecasting engine, we need to gather our toolkit. These libraries act like the crew in a command center—each with a clear role:

pandas handles our data logistics.

tensorflow (for later graphing and experimentation) brings deep learning muscle if we need it.

matplotlib.pyplot gives us the radar screen—essential for visual diagnostics.

re and sys help us parse filenames and handle command-line operations like a proper data ops terminal.

```
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import sys
```

---

🔹 Step 2: Load and Preprocess Market Data

- Accepts a CSV filename from command line arguments  
- Extracts the date from the filename for logging and versioning  
- Reverses data to maintain chronological order  
- Reduces dataset to the latest 10,000 records  
- Adds engineered features: `Future_Median`, price change %, rolling volume averages, lagged medians

---

🔹 Step 3: Feature Selection

We prepare the feature matrix `X` and target variable `y`:

```
features = ["Open", "High", "Low", "Close", "Price_Difference", "Open_Close_Change_Pct",
            "High_Low_Change_Pct", "Volume", "Volume_MA_20", "Volume_Change_Pct",
            "median_t-1", "median_t-2"]
```

Missing values are removed to ensure consistency.

---

🔹 Step 4: Scaling and Splitting

We scale the data using `StandardScaler` and split into training and testing sets:

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

🔹 Step 5: Train SVR and SFN with Grid Search

**LinearSVR:**

```
param_grid = {'C': [1, 10, 50], 'epsilon': [0.01, 0.1, 1]}
```

**SFN (MLPRegressor):**

```
param_grid_sfn = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}
```

---

🔹 Step 6: Ensemble the Models

We use `VotingRegressor` to combine SVR and SFN:

```
ensemble_model = VotingRegressor(estimators=[('linear_svr', best_linear_svr_model),
                                             ('sfn', best_sfn_model)],
                                 weights=[0.7, 0.3])
ensemble_model.fit(X_train, y_train)
```

---

🔹 Step 7: Evaluate Model Performance

We compute metrics like MSE, RMSE, MAE, R², and explained variance.  
Visual comparisons are plotted for training and testing data.

---

🔹 Step 8: Save Logs and Export to ONNX

- Model logs are saved in JSON format  
- Includes feature names, scaling parameters, performance metrics, and model name/version

We then export the trained model to ONNX format:

```
from skl2onnx import convert_sklearn
onnx_model = convert_sklearn(ensemble_model, initial_types=[('input', FloatTensorType([None, X_test.shape[1]]))])
```

---

🔹 Step 9: Send Model Log to Arasaka Neural Bastion (Optional API Endpoint)

If online, we send the performance and meta data to a remote backend for storage and monitoring:

```
result = SendToArasaka(log_data)
print(result)
```

---

✅ Summary

This ensemble model is part of an ongoing exploration of hybrid architectures for time-series forecasting.  
Exporting to ONNX also ensures compatibility with non-Python runtime environments.

Stay tuned for part 2: **Feature Selection and Hyperparameter Tuning Techniques**.

