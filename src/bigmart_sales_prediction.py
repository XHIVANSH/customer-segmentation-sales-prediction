"""
bigmart_sales_prediction.py

Big Mart Sales Prediction using regression models.

Pipeline:
  1. Load data & handle missing values
  2. Clean inconsistent categorical labels (e.g. Item_Fat_Content typos)
  3. Feature engineering (outlet age, item categories, etc.)
  4. Preprocessing pipeline: scaling (numeric) + one-hot encoding (categorical)
  5. Train & compare multiple regressors
  6. Hyperparameter tuning (GridSearchCV) on the best-performing model
  7. Evaluate with RMSE / MAE / R^2 and save results + plots

Usage:
    python src/bigmart_sales_prediction.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

sns.set_theme(style="whitegrid")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "bigmart_train.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"

TARGET = "Item_Outlet_Sales"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {path.name}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize inconsistent Item_Fat_Content labels
    fat_map = {
        "low fat": "Low Fat",
        "LF": "Low Fat",
        "Low Fat": "Low Fat",
        "reg": "Regular",
        "Regular": "Regular",
    }
    df["Item_Fat_Content"] = df["Item_Fat_Content"].map(fat_map).fillna(df["Item_Fat_Content"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Outlet age relative to a fixed reference year (dataset convention: 2013)
    df["Outlet_Age"] = 2013 - df["Outlet_Establishment_Year"]

    # Broad item category from the identifier prefix (FD=Food, DR=Drinks, NC=Non-Consumable)
    df["Item_Category"] = (
        df["Item_Identifier"].str[:2].map({"FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"})
    )

    # Non-consumables shouldn't really have a fat content label
    df.loc[df["Item_Category"] == "Non-Consumable", "Item_Fat_Content"] = "Non-Edible"

    df = df.drop(columns=["Item_Identifier", "Outlet_Establishment_Year"])
    return df


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def evaluate(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    print(f"{name:22s} | RMSE: {rmse:8.2f} | MAE: {mae:8.2f} | R^2: {r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def main():
    parser = argparse.ArgumentParser(description="Big Mart Sales Prediction")
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to bigmart_train.csv"
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_data(Path(args.data))
    df = clean_data(df)
    df = engineer_features(df)

    print("\nMissing values after cleaning:\n", df.isnull().sum())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # Everything that isn't numeric is treated as categorical. This avoids
    # select_dtypes(include=["object", "str"]) directly, whose behavior
    # differs across pandas versions (some raise a TypeError on that combo).
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    print(f"\nNumeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200),
    }

    print("\n--- Baseline Model Comparison ---")
    results = []
    fitted_pipelines = {}
    for name, estimator in candidates.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        results.append(evaluate(pipe, X_test, y_test, name))

    results_df = pd.DataFrame(results).sort_values("rmse")
    best_model_name = results_df.iloc[0]["model"]
    print(f"\nBest baseline model: {best_model_name}")

    # --- Hyperparameter tuning on the best model family ---
    print(f"\n--- Hyperparameter Tuning: {best_model_name} ---")
    if best_model_name == "Random Forest":
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 8, 12, 16],
            "model__min_samples_leaf": [1, 2, 4],
        }
        base_estimator = RandomForestRegressor(random_state=42)
    elif best_model_name == "Decision Tree":
        param_grid = {
            "model__max_depth": [4, 6, 8, 10, None],
            "model__min_samples_leaf": [1, 2, 4, 8],
        }
        base_estimator = DecisionTreeRegressor(random_state=42)
    else:
        param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0]}
        base_estimator = Ridge(random_state=42)

    tuned_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", base_estimator)])
    grid_search = GridSearchCV(
        tuned_pipe, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best params: {grid_search.best_params_}")
    tuned_result = evaluate(grid_search.best_estimator_, X_test, y_test, f"Tuned {best_model_name}")
    results.append(tuned_result)

    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print("\n--- Final Results Summary ---")
    print(results_df.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "bigmart_model_results.csv", index=False)

    # Plot: model comparison
    plt.figure(figsize=(9, 5))
    sns.barplot(data=results_df, x="model", y="rmse", hue="model", palette="viridis", legend=False)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("RMSE (lower is better)")
    plt.title("Model Comparison — Big Mart Sales Prediction")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bigmart_model_comparison.png", dpi=150)
    plt.close()

    # Plot: predicted vs actual for the final tuned model
    final_preds = grid_search.best_estimator_.predict(X_test)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, final_preds, alpha=0.4, s=18, color="#4C72B0")
    lims = [min(y_test.min(), final_preds.min()), max(y_test.max(), final_preds.max())]
    plt.plot(lims, lims, "r--", linewidth=1.5, label="Ideal fit")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title(f"Predicted vs Actual — Tuned {best_model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bigmart_predicted_vs_actual.png", dpi=150)
    plt.close()

    print(f"\nSaved results & plots -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
