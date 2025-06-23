---

title: "Teaching Machines to See: SURF, HOG & Bagging Magic in Image Classification"
date: 2025-06-23
tags: \[Computer Vision, Machine Learning, Python, Ensemble Learning, Image Processing]
categories: \[Projects]
description: "A walkthrough of building an image classifier using SURF, HOG features and Bagging, with Decision Trees and Random Forests in the mix."
-----------------------------------------------------------------------------------------------------------

What if we could teach machines to see—not just glance, but *perceive*, understand, and categorize? In this post, we step into the world of image classification using a blend of classical computer vision and ensemble machine learning. Picture it like giving a robot eyes (SURF + HOG) and a brain (Bagging + Decision Trees) to recognize visual patterns.

---

**🔹 Step 1: Image Data Mounting and Exploration**

Using Google Colab, we mount a Drive folder containing labeled subfolders of images. Each subfolder is a class: cats, dogs, raccoons—you name it.

```python
drive.mount('/content/drive')
```

We then peek into image dimensions to get a feel for our dataset's scale and variety.

---

**🔹 Step 2: Feature Extraction – Meet SURF and HOG**

Imagine each image as a complex terrain map. SURF (or fallback SIFT) identifies critical landforms—edges, corners, and blobs. HOG (Histogram of Oriented Gradients) captures the overall texture and flow of the terrain.

Each image goes through:

* Grayscale conversion
* SURF/SIFT descriptor flattening (max 1000 features)
* HOG feature extraction
* Feature vector concatenation

These vectors are then padded for consistent dimensions.

---

**🔹 Step 3: Label Encoding**

To make our labels machine-readable, we use `LabelEncoder`:

```python
labelencoder = LabelEncoder()
enc_labels = labelencoder.fit_transform(labels)
```

This allows us to train classifiers without worrying about string categories.

---

**🔹 Step 4: Ensemble Learning with Bagging**

We train a `BaggingClassifier` using `DecisionTreeClassifier` as the base estimator. It's like assembling a team of scouts, each examining different subsets of the data and reporting back.

```python
bagging_classifier = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    max_samples=0.8,
    random_state=42
)
```

Split the data into training and testing:

```python
X_train, X_test, y_train, y_test = train_test_split(images, enc_labels, test_size=0.2, random_state=42)
bagging_classifier.fit(X_train, y_train)
```

---

**🔹 Step 5: Classical Decision Trees and Random Forests**

We benchmark our ensemble with standalone classifiers:

```python
DecisionTreeClassifier(criterion='entropy')
RandomForestClassifier(max_depth=7, random_state=0)
```

Each tree is like a rulebook; the forest is a democracy of rulebooks. Surprisingly, forests often outperform individual trees due to their aggregated wisdom.

---

**🔹 Step 6: Evaluation and Confusion Matrix**

Using `classification_report` and `confusion_matrix`, we assess precision, recall, and accuracy. Visual heatmaps reveal which classes are confused with others.

```python
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
```

---

**🔹 Step 7: Dataset Summary**

We count how many images each class has:

```python
image_counts = {folder: len(glob.glob(os.path.join(folder_path, "*.jpg"))) for folder in subfolders}
```

This reveals any class imbalance issues.

---

**🔹 Step 8: Fast Linear Classifier – SGD**

For a lightweight baseline, we deploy an `SGDClassifier` using a linear hinge loss. Surprisingly robust and fast, SGD often serves as a sanity check for larger models.

```python
sgd_classifier = SGDClassifier(loss='hinge', penalty='l2')
```

---

**📌 Summary**

In this journey, we stitched classical computer vision (SURF + HOG) with robust ensemble learning (Bagging, Forests, SGD). It’s a practical example of fusing domain-specific handcrafted features with reliable ML architectures.

By the end, we have:

* Extracted meaningful visual descriptors
* Trained multiple classifiers
* Evaluated models using structured metrics
* Export-ready performance metrics for deployment

**Coming Next:** Pushing this into a real-world application with live image streams, data augmentation, and model compression for edge devices.
