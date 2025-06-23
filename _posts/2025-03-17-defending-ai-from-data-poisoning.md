---
title: "Cybersecurity and AI Case Studies Project"
layout: single
classes: wide
author_profile: true
read_time: true
comments: true
share: true
toc: true
toc_label: "Project Phases"
toc_sticky: true
---

This project focuses on AI-based intrusion detection using the Edge-IIoTset dataset, exploring various stages such as preprocessing, model training, data poisoning simulation, and mitigation strategies.

## 📁 Google Colab Python Script

# 💾 Phase 1: Data Extraction & Preprocessing

Jack in Edge-IIoTset dataset into Google Colab. No manual Dowloads, chooms. Right, switch to GPU. Trust me.

```python
from google.colab import files
!pip install -q kaggle
```

Stay sharp, we are about to:
1. Automate dataset fetching from Kaggle. (Dowloading it on mobile data? No choom. We're not doing that.)
2. Set up API keys securely in Colab.
3. Unzip and remove junk (because we keep it clean - nice and proper-like).

Upload API key. Kaggle API key. It's on Kaggle. Tinker with it choom, you'll figure it out.
```python
files.upload()

# Setup Kaggle credentials
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

#!/bin/bash

# Download the dataset
!kaggle datasets download -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot --unzip -p ./EdgeIIoT-dataset

# !kaggle datasets download -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot -f "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
```
**Checkpoint:** Load and Explore the Data.

Let's see if we'll need to oversample something. Plus I'm curious about how this data looks like at a glance.
```python

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('/content/EdgeIIoT-dataset/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv', low_memory=False)

df.head(5)

# Attack type distribution
print(df['Attack_type'].value_counts())

```
# 🧹**Phase 2:** Data Cleanup & Preprocessing

Feed the machine! Nah, the machine has dietary requirements. Junk-intolerant

```python

from sklearn.utils import shuffle

# Drop redundant & irrelevant columns
drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
                "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
                "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
                "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"]

df.drop(drop_columns, axis=1, inplace=True)

# Remove missing values & duplicates
df.dropna(axis=0, how='any', inplace=True)
df.drop_duplicates(subset=None, keep="first", inplace=True)

# Shuffle the deck to keep training random
df = shuffle(df)

# Check if we cleaned everything
df.isna().sum()
print(df['Attack_type'].value_counts())
```

1. We've stripped non-essential columns.
2. Scrubbed missing or duplicate rows.
3. Shuffled to prevent order-bias.

Encode Categorical Data
1. One-hot encoding lets the Machine digest categorical features properly. Special diets.

```python

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# Function to encode categorical features
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])  # Convert categorical values into one-hot encoding
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

# Apply encoding to necessary columns
encode_text_dummy(df, 'http.request.method')
encode_text_dummy(df, 'http.referer')
encode_text_dummy(df, "http.request.version")
encode_text_dummy(df, "dns.qry.name.len")
encode_text_dummy(df, "mqtt.conack.flags")
encode_text_dummy(df, "mqtt.protoname")
encode_text_dummy(df, "mqtt.topic")

df['Attack_type'].value_counts()

```

# **📦 Phase 3: Save Preprocessed Dataset**

I'm keeping things modular—compartmentalized. I really don't want to do the whole API gig on every run choom.
I guess we are ready to feed the machine.

```python

df.to_csv('preprocessed_DNN.csv', encoding='utf-8')

df.head(5)

```

# ⚔️ **Phase 4: AI Model Development**

Train the Intrusion Detection AI Model
1. Random Forest for first-pass evaluation.
2. Train-test split: 80/20 ratio.
3. Performance report with classification metrics.

```python

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed dataset
df = pd.read_csv('preprocessed_DNN.csv')

# Split into features and labels
X = df.drop("Attack_type", axis=1)  # Features
y = df["Attack_type"]  # Labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import matplotlib.pyplot as plt

labels = ['Train', 'Test']
sizes = [0.8, 0.2]  # 80% train, 20% test
colors = ['lightblue', 'lightcoral']
explode = (0.1, 0)  # Explode the 'Train' slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')

# Set the title
plt.title('Train-Test Split')

# Show the plot
plt.show()

# Initialize and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

```

# **💀 Phase 5: Data Poisoning**

Simulating an Insider Threat Attack

```python

def poison_data(X, y, poison_fraction=0.1):
    """Injects label-flipped samples into the dataset."""
    num_poison = int(len(y) * poison_fraction)
    poison_indices = np.random.choice(len(y), num_poison, replace=False)
    y_poisoned = y.copy()
    y_poisoned.iloc[poison_indices] = np.random.choice(y.unique(), size=num_poison)  # Flip labels randomly
    return X, y_poisoned

def detect_anomalies(X, contamination=0.1):
    """Detects potential poisoned samples using Isolation Forest."""
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    preds = iso_forest.fit_predict(X)
    return preds == 1  # Returns a boolean mask of non-anomalous samples
```

From here we test the accuracy before and after poisoning. Then we impliment a way to detect anomalies in the data.

```python

# Load preprocessed dataset
df = pd.read_csv('preprocessed_DNN.csv')
# Split into features and labels
X = df.drop("Attack_type", axis=1)  # Features
y = df["Attack_type"]  # Labels

# Poison the dataset
X_poisoned, y_poisoned = poison_data(X, y, poison_fraction=0.01)

# Detect and remove anomalies
clean_mask = detect_anomalies(X_poisoned)
X_cleaned, y_cleaned = X_poisoned[clean_mask], y_poisoned[clean_mask]

```
Split the poisoned then cleaned dataset

```python
# Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_poisoned, y_poisoned, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate after anomaly detection and correction
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

```

The difference between these two cells are just the comments. I'm sure i can come up with something better but not now. I'll probably do that in the second draft, choom.

```python

# Evaluate With poisoned data
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

```
The next is evaluate with clean, untouched data

```python

# Clean data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed dataset
df = pd.read_csv('preprocessed_DNN.csv')

# Split into features and labels
X = df.drop("Attack_type", axis=1)  # Features
y = df["Attack_type"]  # Labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
```

So far we've just Simulated a real-world poisoning attack.
Next is:
Measuring accuracy degradation as poison % increases

```python

def evaluate_model(X, y, poison_fraction):
    """Evaluates model accuracy before and after data poisoning and anomaly detection."""
    X_poisoned, y_poisoned = poison_data(X, y, poison_fraction)

    # Train-test split before defense
    X_train, X_test, y_train, y_test = train_test_split(X_poisoned, y_poisoned, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    poisoned_acc = accuracy_score(y_test, y_pred)

    # Apply defense mechanism
    clean_mask = detect_anomalies(X_poisoned)
    X_cleaned, y_cleaned = X_poisoned[clean_mask], y_poisoned[clean_mask]

    # Train-test split after defense
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cleaned_acc = accuracy_score(y_test, y_pred)

    return poisoned_acc, cleaned_acc

import matplotlib.pyplot as plt

poison_fractions = np.linspace(0, 1, 10)  # Different levels of poisoning
poisoned_accuracies = []
cleaned_accuracies = []

for fraction in poison_fractions:
    poisoned_acc, cleaned_acc = evaluate_model(X, y, fraction)
    poisoned_accuracies.append(poisoned_acc)
    cleaned_accuracies.append(cleaned_acc)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(poison_fractions, poisoned_accuracies, label="Before Defense", marker='o', linestyle='dashed', color='r')
plt.plot(poison_fractions, cleaned_accuracies, label="After Defense", marker='s', linestyle='solid', color='g')
plt.xlabel("Poisoning Fraction")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Before and After Defense Against Data Poisoning")
plt.legend()
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Bar Plot Visualization
bar_width = 0.35  # Width of each bar
index = np.arange(len(poison_fractions))  # The x locations for the groups

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(index - bar_width / 2, poisoned_accuracies, bar_width, label='Before Defense', color='r')
rects2 = ax.bar(index + bar_width / 2, cleaned_accuracies, bar_width, label='After Defense', color='g')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Poisoning Fraction', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Accuracy Before and After Defense Against Data Poisoning', fontsize=14, fontweight='bold')
ax.set_xticks(index)
ax.set_xticklabels(poison_fractions)
ax.legend(fontsize=10)

plt.tight_layout()
plt.show()
```

# 🔐 Phase 6: Mitigating Data Poisoning

# 📡 Security Briefing: Locking Down the Data Pipeline
Alright, choom. Time to wire up defenses against data poisoning before some rat flips our AI into thinking attacks are safe traffic. We don't want that so here's my game plan:
1. **Cryptographic Hashing: Integrity checkpoints.**
I'll slap a digital fingerprint on the dataset. If some gonk tampers with the data, the fingerprint changes—instant red flag.
Every dataset version gets logged. Before training, we check its hash—if it doesn't match the last known clean version, we know someone messed with it.
Why it works? I don't know choom. It's the same tech that corpos use so it must be better than junk off the streets.


2. **Anomaly Detection: Sniffing Out Poisoned Labels**

🔹 What it does: Scans for suspicious label shifts before training. If attackers flip enough labels to change the dataset’s DNA, this system flags it.

🔹 How we run it: Isolation Forest scans label distributions in real-time. If malicious traffic suddenly looks too "benign", it sounds the alarm.

🔹 Why it works: We’re catching the attack before it trains the AI wrong—like detecting a virus before it spreads.

3. **Consensus Learning: Cross-Check Models**

🔹 What it does: Instead of trusting one AI model, we train multiple models and compare their predictions. If one model starts acting sketchy (due to poisoned training), we override it.

🔹 How we run it: Majority vote—if 3 models flag a traffic packet as an attack and 1 calls it "benign," we trust the majority.

🔹 Why it works: Attackers would have to poison multiple models at once to break the system—way harder to pull off.

# **Real-Time Defense Against Label Poisoning**

We'll implement two defense mechanisms:
1. Anomaly Detection on Labels (detect poisoned labels before training).

2. Blockchain-Based Integrity Check (log dataset hashes to prevent tampering).

⚡ 1️⃣ Detect Label Manipulation with Anomaly Detection

📌 What this does:

Before training, we analyze label distributions.
If the labels suddenly shift (e.g., too many benign samples injected), we flag an anomaly.
Uses Isolation Forest (an unsupervised anomaly detection model).

**🔧 Code: Anomaly Detection for Poisoned Labels**

```python

from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

# Function to detect label poisoning
def detect_label_poisoning(y_train):
    y_values = y_train.value_counts(normalize=True).values.reshape(-1, 1)  # Normalize label distribution

    iso_forest = IsolationForest(contamination=0.1, random_state=42)  # 10% contamination threshold
    iso_forest.fit(y_values)

    anomalies = iso_forest.predict(y_values)

    # Flag if anomaly detected
    if -1 in anomalies:
        print("⚠️ WARNING: Possible label poisoning detected!")
    else:
        print("✅ Label distribution looks normal.")

    return anomalies

# we'll call the function on our Y_Train and we should be good.

```

📌 How this works:
✅ Checks if label frequencies are shifting (e.g., someone flipped too many malicious labels).
✅ Flags anomalies before training even starts.
✅ Works in real-time before model training.

```python

df = pd.read_csv('preprocessed_DNN.csv')

# Split into features and labels
X = df.drop("Attack_type", axis=1)  # Features
y = df["Attack_type"]  # Labels

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

anomalies = detect_label_poisoning(y_train)
print(anomalies)

```

# 🔗 2️⃣ **Blockchain-Based Dataset Integrity Check**
📌 What this does:

Every time you modify the dataset, its hash is stored.
Before training, we recompute the hash and compare it to previous versions.
If any data is poisoned, the hash won't match, and we raise an alarm.

🔧 **Code: Log Dataset Integrity with SHA-256 Hashes**

```python

import hashlib
import json

# Function to compute dataset hash
def compute_hash(df):
    df_str = df.to_json()  # Convert DataFrame to JSON string
    return hashlib.sha256(df_str.encode()).hexdigest()  # Generate SHA-256 hash

# Simulated blockchain ledger (stores dataset history)
dataset_ledger = {}

# Log dataset state
def log_dataset_version(df, version="initial"):
    dataset_hash = compute_hash(df)
    dataset_ledger[version] = dataset_hash
    print(f"✅ Dataset version {version} logged. Hash: {dataset_hash}")

# Check dataset integrity
def check_dataset_integrity(df, version="initial"):
    current_hash = compute_hash(df)
    if dataset_ledger.get(version) == current_hash:
        print(f"✅ Dataset integrity verified (Version: {version}). No tampering detected.")
    else:
        print(f"⚠️ WARNING: Dataset tampering detected! (Version: {version})")

# Example Usage
log_dataset_version(df, "clean")  # Store clean dataset hash
# Simulate dataset modification (poisoning)
df.iloc[0, -1] = "Benign"  # Flip label
check_dataset_integrity(df, "clean")  # Compare new hash to original
```