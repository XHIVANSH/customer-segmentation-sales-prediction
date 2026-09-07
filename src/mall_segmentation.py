"""
mall_segmentation.py

Mall Customer Segmentation using K-Means Clustering.

Pipeline:
  1. Load & explore the data
  2. Feature scaling (StandardScaler)
  3. Determine optimal k via the Elbow Method + Silhouette Score
  4. Fit final K-Means model
  5. Profile & visualize the resulting customer segments

Usage:
    python src/mall_segmentation.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "Mall_Customers.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {path.name}")
    return df


def explore_data(df: pd.DataFrame):
    print("\n--- Data Overview ---")
    print(df.describe(include="all").T)
    print("\nMissing values:\n", df.isnull().sum())


def select_features(df: pd.DataFrame, feature_cols):
    return df[feature_cols].copy()


def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 11), out_dir: Path = OUTPUT_DIR):
    """Elbow method (inertia) + silhouette score across a range of k."""
    inertias, sil_scores = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(list(k_range), inertias, marker="o", color="#4C72B0")
    axes[0].set_title("Elbow Method (Inertia vs k)")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia (WCSS)")

    axes[1].plot(list(k_range), sil_scores, marker="o", color="#DD8452")
    axes[1].set_title("Silhouette Score vs k")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "mall_k_selection.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved k-selection plot -> {fig_path}")

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"Best k by silhouette score: {best_k} (score={max(sil_scores):.3f})")
    return best_k, inertias, sil_scores


def fit_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    model.fit(X_scaled)
    return model


def profile_clusters(df: pd.DataFrame, feature_cols, cluster_labels) -> pd.DataFrame:
    df = df.copy()
    df["Cluster"] = cluster_labels
    profile = df.groupby("Cluster")[feature_cols].mean().round(1)
    profile["Count"] = df.groupby("Cluster").size()
    print("\n--- Cluster Profiles (mean feature values) ---")
    print(profile)
    return df


def plot_clusters_2d(df: pd.DataFrame, x_col: str, y_col: str, out_dir: Path = OUTPUT_DIR):
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=df["Cluster"].nunique())
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue="Cluster",
        palette=palette,
        s=70,
        alpha=0.85,
        edgecolor="white",
    )
    plt.title(f"Customer Segments: {x_col} vs {y_col}")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "mall_customer_clusters.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved cluster scatter plot -> {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Mall Customer Segmentation via K-Means")
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to Mall_Customers.csv"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Force a specific number of clusters (skips auto-selection)",
    )
    args = parser.parse_args()

    df = load_data(Path(args.data))
    explore_data(df)

    feature_cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = select_features(df, feature_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if args.k is None:
        best_k, _, _ = find_optimal_k(X_scaled)
    else:
        best_k = args.k

    model = fit_kmeans(X_scaled, best_k)
    labeled_df = profile_clusters(df, feature_cols, model.labels_)

    plot_clusters_2d(labeled_df, feature_cols[0], feature_cols[1])

    out_csv = OUTPUT_DIR / "mall_customers_segmented.csv"
    labeled_df.to_csv(out_csv, index=False)
    print(f"\nSaved labeled dataset -> {out_csv}")


if __name__ == "__main__":
    main()
