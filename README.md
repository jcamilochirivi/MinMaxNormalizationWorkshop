# 🏠 Min-Max Normalization & PCA Workshop

## 📘 Workshop Overview

This workshop introduces students to two foundational techniques in Machine Learning preprocessing and dimensionality reduction:

- **Min-Max Normalization**
- **Principal Component Analysis (PCA)**

Using a real-world **housing dataset**, students will first normalize all numerical features to ensure fair comparison and proper scaling. They will then extend their analysis by applying PCA to identify the most important underlying structure in the data.

This workshop is designed for active learning, collaborative coding, and reflective thinking.

---

## 🧠 Topics Covered

### 1️⃣ Min-Max Normalization
- Linear feature scaling
- Implementation from scratch
- Handling edge cases (e.g., constant columns)
- Why normalization matters for ML algorithms (KNN, regression, neural networks)

### 2️⃣ Principal Component Analysis (PCA)
- Variance maximization
- Eigenvectors and eigenvalues (conceptual level)
- Dimensionality reduction
- Interpreting principal components
- Relating principal components to a target variable

---

## 🎯 Expected Learning Outcomes

By the end of this workshop, students should be able to:

- Implement Min-Max normalization across all numerical features of a dataset.
- Explain why feature scaling is necessary before applying PCA.
- Apply PCA to a normalized housing dataset.
- Determine how many principal components explain ≥ 90% of the variance.
- Interpret feature loadings and relate principal components to a target variable.
- Critically evaluate code implementation through structured “talking points.”

---

## 🧩 Summary of Activity

### Step 1: Normalize the Housing Dataset
Students implement Min-Max normalization manually on all relevant numerical features.

### Step 2: Explore Principal Components
Students apply PCA to the normalized feature matrix to:
- Identify major variance directions
- Determine cumulative explained variance
- Analyze feature loadings
- Examine correlation between principal components and the target variable

### Step 3: Reflection & Talking Points
Students document three “talking points” discussing:
- Implementation decisions
- Efficiency or logic considerations
- Interpretation of PCA results

---

## 📂 Repository Contents

- `MinMax_Normalization_Workshop_vNext_PCA.ipynb`
- `housing_data.csv`
- `README.md` (this file)

---

## 🚀 Why This Matters

Feature scaling and dimensionality reduction are critical skills in modern Machine Learning workflows. Understanding both the mechanics and interpretation of these techniques prepares students for:

- Model optimization
- Feature engineering
- Data preprocessing pipelines
- Real-world ML deployment challenges

---


