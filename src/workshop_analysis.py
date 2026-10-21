"""Reusable object-oriented code for the housing Min-Max and PCA workshop.

Min-Max is implemented manually with pandas. No NumPy or scikit-learn
normalizer is used. PCA itself is performed with scikit-learn.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class HousingDataset:
    """Read and inspect the supplied housing CSV without creating fake data."""

    def __init__(self, csv_path="housing_data.csv"):
        self.csv_path = Path(csv_path)
        self.data = None

    def load(self):
        """Load the housing data from a local CSV file."""
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"Housing dataset not found: {self.csv_path}. "
                "Keep housing_data.csv next to the notebook."
            )
        self.data = pd.read_csv(self.csv_path)
        return self.data.copy()

    def quality_report(self):
        """Return basic checks we can discuss during the presentation."""
        if self.data is None:
            raise ValueError("Load the dataset before checking its quality.")
        return pd.DataFrame({
            "dtype": self.data.dtypes.astype(str),
            "missing_values": self.data.isna().sum(),
            "unique_values": self.data.nunique(),
        })


class ManualMinMaxNormalizer:
    """Scale chosen numerical columns to [0, 1] using the original formula.

    We keep the original columns so we can later compare raw and normalized data.
    A constant column is assigned 0.0 to avoid dividing by zero.
    """

    def __init__(self, columns):
        self.columns = list(columns)
        self.minimums = {}
        self.maximums = {}
        self.scaled_data = None

    def fit_transform(self, data):
        """Calculate each column's min/max and add a _norm copy."""
        result = data.copy()
        for column in self.columns:
            if column not in result.columns:
                raise KeyError(f"Missing feature: {column}")
            if not pd.api.types.is_numeric_dtype(result[column]):
                raise TypeError(f"Expected numerical values in {column}")
            if result[column].isna().any():
                raise ValueError(f"Missing values in {column}; clean them first")
            minimum = result[column].min()
            maximum = result[column].max()
            self.minimums[column] = minimum
            self.maximums[column] = maximum
            if maximum == minimum:
                # All houses share the same value; there is no range to scale.
                result[f"{column}_norm"] = 0.0
            else:
                result[f"{column}_norm"] = (
                    result[column] - minimum
                ) / (maximum - minimum)
        self.scaled_data = result
        return result.copy()

    def comparison(self, rows=5):
        """Show original values beside their scaled versions."""
        if self.scaled_data is None:
            raise ValueError("Normalize the dataset before making a comparison.")
        columns = []
        for feature in self.columns:
            columns.extend([feature, f"{feature}_norm"])
        return self.scaled_data[columns].head(rows)

    def validate(self):
        """Check that every normalized feature has values between 0 and 1."""
        if self.scaled_data is None:
            raise ValueError("Normalize the dataset before validation.")
        checks = {}
        for feature in self.columns:
            values = self.scaled_data[f"{feature}_norm"]
            checks[feature] = bool(values.notna().all() and values.between(0, 1).all())
        return pd.DataFrame.from_dict(checks, orient="index", columns=["in_range_0_to_1"])


class HousingPCAAnalysis:
    """Perform PCA on already-normalized predictor features only."""

    def __init__(self, features, target="Price", threshold=0.90):
        self.features = list(features)
        self.target = target
        self.threshold = threshold
        self.pca = None
        self.scores = None
        self.target_values = None
        self.feature_matrix = None

    def fit(self, data):
        """Find PCA directions and save each house's principal-component scores."""
        if self.target in self.features or f"{self.target}_norm" in self.features:
            raise ValueError("Do not include the Price target as a PCA predictor.")
        columns = [f"{feature}_norm" for feature in self.features]
        missing = [column for column in columns + [self.target] if column not in data]
        if missing:
            raise KeyError(f"Required columns are missing: {missing}")
        self.feature_matrix = data[columns].copy()
        if self.feature_matrix.isna().any().any():
            raise ValueError("PCA predictors contain missing values")
        self.target_values = data[self.target].copy()
        self.pca = PCA()
        array = self.pca.fit_transform(self.feature_matrix)
        names = [f"PC{n + 1}" for n in range(array.shape[1])]
        self.scores = pd.DataFrame(array, columns=names, index=data.index)
        return self.scores.copy()

    def _ensure_fitted(self):
        if self.pca is None:
            raise ValueError("Call fit() before asking for PCA results.")

    def variance_table(self):
        """Report individual and cumulative explained variance for each PC."""
        self._ensure_fitted()
        fractions = self.pca.explained_variance_ratio_
        return pd.DataFrame({
            "component": self.scores.columns,
            "explained_variance_ratio": fractions,
            "cumulative_variance_ratio": np.cumsum(fractions),
        })

    def components_needed(self):
        """Smallest number of PCs reaching our target variance (normally 90%)."""
        fractions = self.variance_table()["cumulative_variance_ratio"].to_numpy()
        return int(np.searchsorted(fractions, self.threshold, side="left") + 1)

    def loading_table(self):
        """Show signed coefficients connecting original features and PCs."""
        self._ensure_fitted()
        return pd.DataFrame(
            self.pca.components_.T,
            index=self.features,
            columns=self.scores.columns,
        )

    def strongest_feature(self, component):
        """Largest absolute coefficient, ignoring whether its sign is positive/negative."""
        return str(self.loading_table()[component].abs().idxmax())

    def correlation_table(self):
        """Check relationships with Price only AFTER PCA is fitted."""
        self._ensure_fitted()
        return pd.DataFrame({
            "component": self.scores.columns,
            "pearson_corr_with_price": [
                self.scores[c].corr(self.target_values) for c in self.scores.columns
            ],
        })

    def most_price_related_pc(self):
        """Choose the PC with the greatest absolute target correlation."""
        table = self.correlation_table()
        pos = table["pearson_corr_with_price"].abs().idxmax()
        return str(table.loc[pos, "component"]), float(table.loc[pos, "pearson_corr_with_price"])


class WorkshopVisualizer:
    """Create presentation-ready PCA plots from the analyzer's results."""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def plot_explained_variance(self):
        """Compare separate-PC variance with the growing cumulative total."""
        table = self.analyzer.variance_table()
        x = np.arange(1, len(table) + 1)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x, table["explained_variance_ratio"] * 100, alpha=0.70, label="Individual PC")
        ax.plot(x, table["cumulative_variance_ratio"] * 100,
                marker="o", linewidth=2, label="Cumulative")
        ax.axhline(100 * self.analyzer.threshold, linestyle="--", color="grey",
                   label=f"{100*self.analyzer.threshold:.0f}% goal")
        ax.set(xticks=x, xticklabels=table["component"],
               xlabel="Principal component", ylabel="Explained variance (%)",
               title="PCA: individual and cumulative explained variance")
        ax.set_ylim(0, 105)
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_loadings(self):
        """Heatmap: larger absolute values mean stronger feature contributions."""
        loadings = self.analyzer.loading_table()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        image = ax.imshow(loadings.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1,
                          aspect="auto")
        ax.set_xticks(range(loadings.shape[1]), loadings.columns)
        ax.set_yticks(range(loadings.shape[0]), loadings.index)
        ax.set_title("PCA loadings: contribution of original housing features")
        fig.colorbar(image, ax=ax, label="Signed coefficient")
        fig.tight_layout()
        return fig

    def plot_pc1_pc2(self):
        """Scatter of PC1 and PC2 colored by original price, NOT scaled price."""
        scores = self.analyzer.scores
        fig, ax = plt.subplots(figsize=(9, 6))
        points = ax.scatter(scores["PC1"], scores["PC2"],
                            c=self.analyzer.target_values,
                            cmap="viridis", alpha=0.65, s=18)
        fig.colorbar(points, ax=ax, label="House price")
        ax.set(xlabel="PC1 score", ylabel="PC2 score",
               title="PC1 vs PC2, colored by house price")
        fig.tight_layout()
        return fig
