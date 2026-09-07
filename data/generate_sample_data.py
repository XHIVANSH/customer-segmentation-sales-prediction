"""
generate_sample_data.py

OPTIONAL FALLBACK. The repo ships with the real datasets already in
data/Mall_Customers.csv and data/bigmart_train.csv (sourced from public
GitHub mirrors of the original Kaggle datasets — see README.md).

This script exists only as a fallback: it generates synthetic data with
the same schema in case you ever want to regenerate/replace the real
files, or need offline test data with a guaranteed structure. It is NOT
required for normal use.

  1. Mall_Customers.csv   (Mall Customer Segmentation Data)
  2. bigmart_train.csv    (Big Mart Sales Prediction - Train.csv)
"""

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
DATA_DIR = Path(__file__).parent


def generate_mall_customers(n=200):
    customer_id = np.arange(1, n + 1)
    gender = RNG.choice(["Male", "Female"], size=n, p=[0.44, 0.56])

    age = RNG.integers(18, 71, size=n)

    # Create a few natural income/spending clusters so K-Means has
    # real structure to find (mirrors the pattern in the real dataset).
    cluster_centers = [(25, 20), (25, 80), (55, 20), (55, 80), (50, 50)]
    n_clusters = len(cluster_centers)
    assigned = RNG.integers(0, n_clusters, size=n)

    income = np.zeros(n)
    spending = np.zeros(n)
    for i in range(n):
        income_center, spend_center = cluster_centers[assigned[i]]
        income[i] = np.clip(RNG.normal(income_center, 8), 15, 140)
        spending[i] = np.clip(RNG.normal(spend_center, 12), 1, 99)

    df = pd.DataFrame(
        {
            "CustomerID": customer_id,
            "Gender": gender,
            "Age": age,
            "Annual Income (k$)": income.round(1),
            "Spending Score (1-100)": spending.round().astype(int),
        }
    )
    return df


def generate_bigmart_sales(n=4000):
    item_identifiers = (
        [f"FD{str(i).zfill(2)}" for i in range(1, 20)]
        + [f"NC{str(i).zfill(2)}" for i in range(1, 10)]
        + [f"DR{str(i).zfill(2)}" for i in range(1, 10)]
    )
    item_id = RNG.choice(item_identifiers, size=n)

    item_weight = np.round(RNG.uniform(4.5, 21.5, size=n), 2)
    # introduce some missingness, matching the real dataset's quirks
    weight_missing_idx = RNG.choice(n, size=int(n * 0.08), replace=False)
    item_weight = item_weight.astype(object)
    item_weight[weight_missing_idx] = np.nan

    fat_content_raw = RNG.choice(
        ["Low Fat", "Regular", "LF", "low fat", "reg"], size=n, p=[0.45, 0.35, 0.08, 0.07, 0.05]
    )

    item_visibility = np.round(np.clip(RNG.exponential(0.05, size=n), 0, 0.3), 5)

    item_types = [
        "Fruits and Vegetables",
        "Snack Foods",
        "Household",
        "Frozen Foods",
        "Dairy",
        "Canned",
        "Baking Goods",
        "Health and Hygiene",
        "Soft Drinks",
        "Meat",
        "Breads",
        "Hard Drinks",
        "Others",
        "Starchy Foods",
        "Breakfast",
        "Seafood",
    ]
    item_type = RNG.choice(item_types, size=n)

    item_mrp = np.round(RNG.uniform(31, 266, size=n), 4)

    outlet_ids = [
        "OUT010",
        "OUT013",
        "OUT017",
        "OUT018",
        "OUT019",
        "OUT027",
        "OUT035",
        "OUT045",
        "OUT046",
        "OUT049",
    ]
    outlet_id = RNG.choice(outlet_ids, size=n)

    outlet_year_map = {oid: int(RNG.integers(1985, 2010)) for oid in outlet_ids}
    outlet_est_year = np.array([outlet_year_map[o] for o in outlet_id])

    outlet_size_map = {
        oid: RNG.choice(["Small", "Medium", "High"], p=[0.4, 0.4, 0.2]) for oid in outlet_ids
    }
    outlet_size = np.array([outlet_size_map[o] for o in outlet_id], dtype=object)
    size_missing_idx = RNG.choice(n, size=int(n * 0.15), replace=False)
    outlet_size[size_missing_idx] = np.nan

    outlet_loc_map = {oid: RNG.choice(["Tier 1", "Tier 2", "Tier 3"]) for oid in outlet_ids}
    outlet_location = np.array([outlet_loc_map[o] for o in outlet_id])

    outlet_type_map = {
        oid: RNG.choice(
            ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"],
            p=[0.55, 0.1, 0.1, 0.25],
        )
        for oid in outlet_ids
    }
    outlet_type = np.array([outlet_type_map[o] for o in outlet_id])

    # Build a sales target with real signal baked in (so regression models
    # actually have something to learn), plus noise.
    base = 200
    mrp_effect = item_mrp * 10
    visibility_effect = -item_visibility * 800
    size_bonus = np.where(outlet_size == "High", 400, np.where(outlet_size == "Medium", 200, 0))
    type_bonus = np.where(
        outlet_type == "Grocery Store", -400, np.where(outlet_type == "Supermarket Type3", 500, 0)
    )
    age_effect = (2013 - outlet_est_year) * 5
    noise = RNG.normal(0, 300, size=n)

    sales = base + mrp_effect + visibility_effect + size_bonus + type_bonus + age_effect + noise
    sales = np.clip(sales, 33, 13000).round(4)

    df = pd.DataFrame(
        {
            "Item_Identifier": item_id,
            "Item_Weight": item_weight,
            "Item_Fat_Content": fat_content_raw,
            "Item_Visibility": item_visibility,
            "Item_Type": item_type,
            "Item_MRP": item_mrp,
            "Outlet_Identifier": outlet_id,
            "Outlet_Establishment_Year": outlet_est_year,
            "Outlet_Size": outlet_size,
            "Outlet_Location_Type": outlet_location,
            "Outlet_Type": outlet_type,
            "Item_Outlet_Sales": sales,
        }
    )
    return df


if __name__ == "__main__":
    mall_df = generate_mall_customers(200)
    mall_path = DATA_DIR / "Mall_Customers.csv"
    mall_df.to_csv(mall_path, index=False)
    print(f"Wrote {len(mall_df)} rows -> {mall_path}")

    bigmart_df = generate_bigmart_sales(4000)
    bigmart_path = DATA_DIR / "bigmart_train.csv"
    bigmart_df.to_csv(bigmart_path, index=False)
    print(f"Wrote {len(bigmart_df)} rows -> {bigmart_path}")
