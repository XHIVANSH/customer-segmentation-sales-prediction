import numpy as np
import pandas as pd
import pytest

from bigmart_sales_prediction import build_preprocessor, clean_data, engineer_features


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame(
        {
            "Item_Identifier": [f"FD{i:02d}" for i in range(n // 2)]
            + [f"DR{i:02d}" for i in range(n // 2)],
            "Item_Weight": rng.uniform(4, 20, size=n),
            "Item_Fat_Content": rng.choice(["Low Fat", "LF", "low fat", "Regular", "reg"], size=n),
            "Item_Visibility": rng.uniform(0, 0.2, size=n),
            "Item_Type": rng.choice(["Snack Foods", "Dairy", "Household"], size=n),
            "Item_MRP": rng.uniform(30, 260, size=n),
            "Outlet_Identifier": rng.choice(["OUT010", "OUT013"], size=n),
            "Outlet_Establishment_Year": rng.choice([1999, 2004, 2009], size=n),
            "Outlet_Size": rng.choice(["Small", "Medium", "High", None], size=n),
            "Outlet_Location_Type": rng.choice(["Tier 1", "Tier 2"], size=n),
            "Outlet_Type": rng.choice(["Supermarket Type1", "Grocery Store"], size=n),
            "Item_Outlet_Sales": rng.uniform(50, 5000, size=n),
        }
    )


def test_clean_data_normalizes_fat_content_labels(raw_df):
    cleaned = clean_data(raw_df)
    assert set(cleaned["Item_Fat_Content"].unique()) <= {"Low Fat", "Regular"}


def test_clean_data_does_not_change_row_count(raw_df):
    cleaned = clean_data(raw_df)
    assert len(cleaned) == len(raw_df)


def test_engineer_features_adds_outlet_age(raw_df):
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    assert "Outlet_Age" in engineered.columns
    # 2013 is the reference year used in feature engineering
    expected_age = 2013 - cleaned["Outlet_Establishment_Year"]
    assert (engineered["Outlet_Age"].values == expected_age.values).all()


def test_engineer_features_derives_item_category(raw_df):
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    assert "Item_Category" in engineered.columns
    assert set(engineered["Item_Category"].unique()) <= {"Food", "Drinks", "Non-Consumable"}


def test_engineer_features_drops_identifier_and_year_columns(raw_df):
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    assert "Item_Identifier" not in engineered.columns
    assert "Outlet_Establishment_Year" not in engineered.columns


def test_non_consumables_get_non_edible_fat_content(raw_df):
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    non_consumables = engineered[engineered["Item_Category"] == "Non-Consumable"]
    if len(non_consumables) > 0:
        assert (non_consumables["Item_Fat_Content"] == "Non-Edible").all()


def test_build_preprocessor_fits_and_transforms_without_error(raw_df):
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)

    X = engineered.drop(columns=["Item_Outlet_Sales"])
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)


def test_build_preprocessor_handles_missing_values(raw_df):
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)

    # Outlet_Size has None values injected by the fixture — make sure
    # the pipeline's imputers handle that without raising.
    X = engineered.drop(columns=["Item_Outlet_Sales"])
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    transformed = preprocessor.fit_transform(X)

    assert not np.isnan(
        transformed.toarray() if hasattr(transformed, "toarray") else transformed
    ).any()
