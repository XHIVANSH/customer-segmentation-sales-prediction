import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from mall_segmentation import (
    find_optimal_k,
    fit_kmeans,
    profile_clusters,
    select_features,
)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "CustomerID": np.arange(1, n + 1),
            "Gender": rng.choice(["Male", "Female"], size=n),
            "Age": rng.integers(18, 70, size=n),
            "Annual Income (k$)": rng.uniform(15, 140, size=n),
            "Spending Score (1-100)": rng.uniform(1, 100, size=n),
        }
    )


def test_select_features_returns_only_requested_columns(sample_df):
    cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = select_features(sample_df, cols)
    assert list(X.columns) == cols
    assert len(X) == len(sample_df)


def test_select_features_does_not_mutate_original(sample_df):
    cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = select_features(sample_df, cols)
    X.iloc[0, 0] = -999
    assert sample_df["Annual Income (k$)"].iloc[0] != -999


def test_find_optimal_k_returns_valid_k_in_range(sample_df):
    cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = select_features(sample_df, cols)
    X_scaled = StandardScaler().fit_transform(X)

    k_range = range(2, 6)
    best_k, inertias, sil_scores = find_optimal_k(X_scaled, k_range=k_range)

    assert best_k in list(k_range)
    assert len(inertias) == len(list(k_range))
    assert len(sil_scores) == len(list(k_range))
    # inertia should generally decrease as k grows
    assert inertias[0] >= inertias[-1]


def test_fit_kmeans_produces_expected_number_of_clusters(sample_df):
    cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = select_features(sample_df, cols)
    X_scaled = StandardScaler().fit_transform(X)

    k = 3
    model = fit_kmeans(X_scaled, k)

    assert model.n_clusters == k
    assert len(model.labels_) == len(sample_df)
    assert set(model.labels_) <= set(range(k))


def test_profile_clusters_adds_cluster_column_and_preserves_length(sample_df):
    cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = select_features(sample_df, cols)
    X_scaled = StandardScaler().fit_transform(X)
    model = fit_kmeans(X_scaled, 3)

    labeled_df = profile_clusters(sample_df, cols, model.labels_)

    assert "Cluster" in labeled_df.columns
    assert len(labeled_df) == len(sample_df)
    assert labeled_df["Cluster"].nunique() <= 3
