---
  
title: "Arasaka Trading Systems: Building an Autonomous Ensemble Trader"
date: 2025-06-17
tags: [Algorithmic Trading, Machine Learning, ONNX, MetaTrader, Automation]
categories: [Projects]
description: "A full walkthrough of the Arasaka Trading Systems architecture   - from raw tick data to live market execution   - blending machine precision with human design."
---

*“In the digital underbelly of the forex markets, algorithms don’t sleep.  
They watch, learn, and act   - faster than any human hand ever could.”*

  

## Introduction: The Birth of Arasaka Trading Systems

**Arasaka Trading Systems** is more than a collection of scripts   - it’s a self-updating, semi-autonomous forecasting organism.  
It was designed to **train, evaluate, deploy, and execute** ensemble-based market models with minimal human intervention.

The pipeline runs across **AWS EC2**, **MetaTrader 5**, and a private monitoring API known as the **Arasaka Neural Bastion**.  
Every Monday, the system runs a demo simulation; by Tuesday, it goes live   - the transition from theory to currency, from signal to stake.

At its core, it blends two AI archetypes:
- A **Linear Support Vector Regressor (SVR)**   - disciplined, deterministic.
- A **Simple Feedforward Network (SFN)**   - adaptive, nonlinear, curious.

Together they predict the **next median candlestick** of the USD/JPY forex pair   - the market’s heartbeat.

  

## System Architecture Overview

Below is the full data flow from market to model to money:

```

[MetaTrader 5] → [Extract Market Data.mq5] → CSV
↓
[Python: ensembled_svr_sfn.py] → Train + Evaluate → Export to ONNX
↓
[run_latest.bat / EC2 Scheduler] → Automate the training cycle
↓
[PrinceSwingEnsembled.mq5] → Demo Trading (AutoMonday)
↓
[AutoTuesdayProdPE.mq5] → Live Execution (AutoTuesday)
↓
[Arasaka Neural Bastion API] → Log models, metrics, scaling params

```

Each component handles one link in the chain.  
Lose synchronization, and the entire feedback loop falls apart.

  

## 🔹 Step 1: Data Extraction   - “The Market Feeders”

The ecosystem starts inside MetaTrader 5, using a custom **MQL5 script** named:

```

Extract Market Data.mq5

```

It captures real-time **USD/JPY** candles and exports them as CSV files into:

```

MQL5/Files/

```

These CSVs are timestamped and later read by the Python model trainer.

> Think of this as the “neural input”   - raw sensory data before cognition.

  

## 🔹 Step 2: Model Training   - “The Thinking Core”

Training occurs inside the file:

```

ensembled_svr_sfn.py

````

This is where the AI *learns*.  
It consumes the market CSV, reverses it chronologically, engineers features, and trains an ensemble of two regressors:

### The Features

| Type | Description |
|    |        --|
| Price_Difference | High–Low candle spread |
| Open_Close_Change_Pct | Percentage movement across the candle |
| High_Low_Change_Pct | Volatility gauge |
| Volume_MA_20 | 20-period moving average of volume |
| Volume_Change_Pct | Momentum of market participation |
| median_t-1 / median_t-2 | Temporal lags capturing short-term memory |

These are transformed, scaled, and split before being fed to the learners.

### The Ensemble Logic

```python
ensemble_model = VotingRegressor(
  estimators=[
    ('linear_svr', best_linear_svr_model),
    ('sfn', best_sfn_model)
  ],
  weights=[0.7, 0.3]
)
````

* **SVR (0.7)** → Handles structural patterns.
* **SFN (0.3)** → Adds subtle nonlinear correction.

The resulting ONNX file represents a unified intelligence, ready to operate independently.

  

## 🔹 Step 3: Deployment   - “When the Machine Walks”

Once the ensemble is trained:

1. **ONNX and JSON log files** are exported to two MetaTrader terminals:

   * `D0E8209F77C8CF37AD8BF550E51FF075` → *Demo environment*
   * `BF06F1F02EEF40C01ADCF1B4EBBF23A9` → *Live environment*

2. **run_latest.bat** automates this process, executing:

   * Data extraction
   * Model training
   * ONNX export
   * Metric reporting

3. AWS EC2 runs the bat script on a scheduled basis:

   * **Monday:** `PrinceSwingEnsembled.mq5` → Simulation run
   * **Tuesday:** `AutoTuesdayProdPE.mq5` → Real trades with capital

The transition between the two marks the switch from *observation* to *engagement*.

  

## 🔹 Step 4: Execution   - “Autonomy in Motion”

Once inside MetaTrader, the **ONNX model** takes command.
The MQL5 expert advisors (`PrinceSwingEnsembled.mq5` and `AutoTuesdayProdPE.mq5`) handle:

* Loading the ONNX model
* Normalizing real-time data with stored scaler parameters
* Making candle-by-candle predictions
* Executing BUY/SELL orders based on predicted medians

Every tick becomes a battlefield decision.
If the forecasted median > current close → long;
else → short.

It’s fast, mechanical, impartial   - yet eerily human in rhythm.

  

## 🔹 Step 5: The Neural Bastion   - “The Archive Mind”

After training, the system sends a log payload to the **Arasaka Neural Bastion**, an API endpoint that archives:

* Model name and version
* Feature set
* Performance metrics (MSE, RMSE, MAE, R²)
* Scaling parameters

```python
url = "https://arasaka-neural-bastion.onrender.com/api/trading-data"
response = requests.post(url, json=log_data)
```

Every model becomes a permanent record   - a digital fossil in the evolution of the trading AI.
Failures are not discarded; they are studied.
This is how the machine learns to survive the market jungle.

  

## Observations from the Field: “When Theory Meets the Market”

In isolation, the ensemble performs well.
But under live-fire (real liquidity, slippage, spread, and broker latency), the precision fades.

Possible causes:

1. **Temporal Drift** – Static models can’t adapt to evolving market regimes.
2. **Feature Myopia** – Limited features ignore volatility clusters, liquidity zones, and order flow asymmetry.
3. **Non-adaptive Thresholds** – No dynamic signal smoothing before execution.
4. **Latency Gap** – Seconds between data export → training → MT5 execution may misalign context.
5. **Scaling Mismatch** – Real-time MQL5 preprocessing may differ slightly from Python’s `StandardScaler`.

The system doesn’t break   - it simply **fails to profit**.
It predicts correctly often enough to impress, but not *strategically* enough to win.

  

## Lessons Learned   - “What the Machine Taught Us”

1. **Predictive accuracy ≠ profitability.**
   Markets are adversarial, not static. Every model’s edge decays.

2. **Real intelligence is adaptive.**
   A static ONNX model is like a samurai frozen mid-swing.
   The next evolution needs *continuous online learning* or adaptive signal reinforcement.

3. **Latency kills.**
   Real-world trading demands synchronization   - data, execution, and feedback within milliseconds.

4. **Data is the true weapon.**
   More diverse indicators, rolling volatility, and order-book derived metrics can feed richer signals into the ensemble.

  

## The Next Evolution

Upcoming upgrades under consideration:

* **Arasaka v3.0:** Add volatility-encoded features (ATR, RSI, VIX correlation)
* **Adaptive retraining agent:** Auto-updates the ONNX model weekly
* **Dynamic risk engine:** Position sizing by model confidence
* **Neural Bastion Dashboard:** Real-time metrics visualization

Arasaka’s endgame isn’t just trading   - it’s *learning how to learn*.
To sense. To adapt. To survive.

  

## Conclusion

The **Arasaka Trading System** represents a fusion of engineering and intuition:
a modular machine built to learn from data and act on its own.
Even when it fails, it does so *intelligently*, leaving a trail of telemetry to refine the next iteration.

*In the end, profit is just a byproduct of understanding.*
The true mission is alignment   - between algorithm and market, between human insight and synthetic precision.

  

**Coming soon → Part II: “Synthetic Instinct: Building the Adaptive Trader.”**

```

  

Would you like me to follow this up with **Part II**   - the one that introduces *Synthetic Instinct*, adaptive retraining, and smarter trade logic   - in the same cyberpunk-teaching tone (essentially the “Arasaka v3.0” chapter)?
```
