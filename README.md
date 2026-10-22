# Min-Max Normalization and PCA Workshop

**Team number:** Group 3
**Team members and student IDs:** Juan Chirivi 9115141 - Ricardo - Zeynep - Senay

## Project Overview

In this workshop, we use a housing dataset to practice two important steps in **feature engineering**: Min-Max normalization and Principal Component Analysis (PCA).

First, we scale numerical housing features to the same range, between **0 and 1**. Then, we apply PCA to understand how information from several features can be represented with fewer principal components. We also check whether the components are related to the house price.

We organized the project using **Object-Oriented Programming (OOP)**. The reusable Python classes are in `workshop_analysis.py`, while the Jupyter notebook shows the workflow, results, plots, and explanations in student-friendly language.

## Dataset

We use the provided `housing_data.csv` file. It contains **2,000 houses and 7 original columns**:

| Column | Description | Use in this workshop |
|---|---|---|
| `House_ID` | Identifier for each house | Identifier only; not used in PCA |
| `Price` | House price | Normalized in Step 1; used as the target when interpreting PCA |
| `Area_sqft` | House area in square feet | Normalization and PCA |
| `Num_Bedrooms` | Number of bedrooms | Normalization and PCA |
| `Num_Bathrooms` | Number of bathrooms | Normalization and PCA |
| `Year_Built` | Year the house was built | Not included in this workshop's PCA analysis |
| `Lot_Size` | Lot size | Normalization and PCA |

Our initial inspection found **no missing values or duplicated rows** in this dataset.


## How to Run the Project

### Requirements

- Python 3
- Jupyter Notebook or VS Code with the Jupyter extension
- Dependencies listed in `requirements.txt`: `pandas`, `numpy`, `matplotlib`, and `scikit-learn`

### Steps

1. Download or clone the project and open its folder in VS Code or Jupyter.
2. Install the dependencies **once from a terminal**, if they are not already installed:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Check that `housing_data.csv` and `workshop_analysis.py` are in the **same folder** as the notebook.
4. Open `MinMax_Normalization_Workshop.ipynb` and select the correct Python kernel.
5. Select **Run All** to execute the notebook from the beginning.

The notebook does **not** contain `%pip install` cells. It reads the local CSV directly and does not create a backup or synthetic dataset.

## Workshop Activities

### Part 1: Min-Max Normalization

We manually normalize five numerical columns:

`Price`, `Area_sqft`, `Num_Bedrooms`, `Num_Bathrooms`, and `Lot_Size`.

The formula is:

```text
Normalized value = (value - minimum) / (maximum - minimum)
```

This transformation puts every selected feature on a scale from **0 to 1**. The class also handles a column with a constant value by assigning 0 instead of dividing by zero.

We implement the calculation with Pandas, **without using Scikit-Learn's MinMaxScaler or a NumPy normalization function**. After normalization, the notebook compares original and normalized values and validates the output ranges.

### Part 2: Principal Component Analysis (PCA)

For PCA, we reuse the four normalized housing predictors:

- `Area_sqft_norm`
- `Num_Bedrooms_norm`
- `Num_Bathrooms_norm`
- `Lot_Size_norm`

We **exclude `Price` from the PCA inputs** because it is the target we want to compare against afterward. PCA is implemented with `sklearn.decomposition.PCA`.

The notebook completes the instructor's **Sections A–F**: choosing the features, checking their scales, fitting PCA, calculating explained variance, interpreting feature loadings, and comparing principal components with the price target.

### Part 3: Visualizations and Talking Points

The notebook includes three visualizations:

1. **Explained variance:** individual and cumulative variance for each principal component, with a 90% reference line.
2. **Feature loadings heatmap:** how strongly each housing feature contributes to each component.
3. **PC1 vs. PC2 scatter plot:** the houses positioned by the first two components and coloured by their original prices.

It also includes **six Talking Points** (three about normalization and three about PCA/implementation choices), with explanations intended to support our five-minute team presentation.

## Results and Interpretation

| Check | Result |
|---|---|
| Number of houses | 2,000 |
| Features normalized | 5 |
| Predictors used in PCA | 4 |
| Components needed for at least 90% explained variance | **3** |
| Cumulative explained variance with 3 components | **Approximately 93.56%** |
| Largest absolute contribution to PC1 | `Num_Bathrooms` |
| Largest absolute contribution to PC2 | `Num_Bedrooms` |
| Strongest observed PC–Price correlation | **PC3: approximately +0.0172** |

**What these results mean:** The first three components retain most of the variation among the four normalized predictors. However, the correlations with `Price` are very close to zero. This means that the main variation captured by PCA does not show a meaningful **linear** relationship with house price in this dataset.

PCA keeps directions with high input variance; it does not automatically find the best predictors of a target variable. Also, normalizing features to 0–1 does not make their variances identical.

## 9. Final Reflection

This workshop helped us understand why feature scaling matters before PCA and how PCA can summarize several related features. Working with OOP also made it easier to separate data loading, preprocessing, analysis, and visualization. Our main takeaway is that **preserving variance is not the same as predicting the target well**: the principal components explain the variation in our housing predictors, but they have very weak linear correlations with house price.
