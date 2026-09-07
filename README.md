# Mall Customer Segmentation & Big Mart Sales Prediction

Two end-to-end machine learning mini-projects in one repo:

1. **Mall Customer Segmentation** — unsupervised K-Means clustering to group mall customers by income and spending behavior.
2. **Big Mart Sales Prediction** — a supervised regression pipeline to forecast per-item sales across retail outlets, with feature engineering, scaling, and hyperparameter tuning.

Both scripts are self-contained, runnable end to end, and produce plots + result tables in `outputs/`.

---

## Project Structure

```
.
├── data/
│   ├── generate_sample_data.py   # generates the synthetic sample CSVs below
│   ├── Mall_Customers.csv        # sample data (mall segmentation)
│   └── bigmart_train.csv         # sample data (sales prediction)
├── src/
│   ├── mall_segmentation.py      # K-Means clustering pipeline
│   └── bigmart_sales_prediction.py  # regression pipeline
├── notebooks/
│   └── exploration.ipynb         # optional interactive EDA notebook
├── tests/
│   ├── test_mall_segmentation.py
│   └── test_bigmart_sales_prediction.py
├── .github/workflows/ci.yml      # lint, test, and pipeline smoke-tests on push/PR
├── outputs/                      # generated plots & result CSVs (created on run)
├── requirements.txt
├── requirements-dev.txt          # + pytest, ruff, black
└── README.md
```

## The data

This repo ships with the **real datasets**, already included in `data/`:

- `data/Mall_Customers.csv` — the original 200-row Mall Customer Segmentation dataset.
- `data/bigmart_train.csv` — the original 8,523-row Big Mart Sales training set (sourced from public mirrors of the well-known Kaggle/Analytics Vidhya datasets, matching the exact schema and values of the canonical files, including their real missing-value pattern and inconsistent `Item_Fat_Content` labels).

Original sources, if you want to verify or re-download them yourself:
- Mall Customer Segmentation Data: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
- Big Mart Sales Prediction (Train.csv): https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data

`data/generate_sample_data.py` is an optional fallback that generates synthetic data with the same schema — useful only if you want to regenerate/replace the bundled files. It isn't needed for normal use.

## Setup

```bash
git clone <this-repo-url>
cd customer-segmentation-sales-prediction
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Regenerate the sample data at any time with:

```bash
python data/generate_sample_data.py
```

---

## 1. Mall Customer Segmentation (K-Means Clustering)

**Goal:** group customers into behaviorally distinct segments (e.g. high income / low spenders, low income / high spenders) using `Annual Income` and `Spending Score`.

**Pipeline:**
- Load & explore the data (nulls, summary stats)
- Standardize features with `StandardScaler`
- Sweep `k = 2..10`, evaluating each with the **Elbow Method** (inertia) and **Silhouette Score**, and auto-select the best `k`
- Fit final `KMeans` model and profile each cluster's average income/spending
- Visualize the segments

**Run it:**

```bash
python src/mall_segmentation.py
# or force a specific number of clusters:
python src/mall_segmentation.py --k 5
```

**Outputs (`outputs/`):**
- `mall_k_selection.png` — elbow + silhouette plots used to pick k
- `mall_customer_clusters.png` — final 2D scatter of segments
- `mall_customers_segmented.csv` — original data with a `Cluster` column appended

**Sample result** (on the real 200-customer dataset, k=5 selected by silhouette score — the well-known 5-cluster "star" pattern for this dataset):

| Cluster | Avg. Income (k$) | Avg. Spending Score | Count | Interpretation |
|---|---|---|---|---|
| 0 | 55.3 | 49.5 | 81 | Mid income, mid spenders (average customers) |
| 1 | 86.5 | 82.1 | 39 | High income, high spenders (target segment) |
| 2 | 25.7 | 79.4 | 22 | Low income, high spenders |
| 3 | 88.2 | 17.1 | 35 | High income, low spenders |
| 4 | 26.3 | 20.9 | 23 | Low income, low spenders |

---

## 2. Big Mart Sales Prediction (Regression)

**Goal:** predict `Item_Outlet_Sales` for each item/outlet combination.

**Pipeline:**
- Load data & clean inconsistent labels (e.g. `LF` / `low fat` / `Low Fat` → one category)
- Feature engineering: `Outlet_Age` (from establishment year), broad `Item_Category` parsed from the item ID prefix
- Preprocessing via `ColumnTransformer`: median imputation + `StandardScaler` for numeric features, most-frequent imputation + `OneHotEncoder` for categorical features — all wrapped in a single `sklearn` `Pipeline` to avoid leakage
- Train & compare **Linear Regression, Ridge, Decision Tree, Random Forest**
- **Hyperparameter tuning** on the best-performing model family via `GridSearchCV` (5-fold CV, RMSE-optimized)
- Evaluate on a held-out test set with RMSE, MAE, R²

**Run it:**

```bash
python src/bigmart_sales_prediction.py
```

**Outputs (`outputs/`):**
- `bigmart_model_results.csv` — metrics for every model tried
- `bigmart_model_comparison.png` — bar chart comparing RMSE across models
- `bigmart_predicted_vs_actual.png` — scatter of predicted vs. actual sales for the final tuned model

**Sample results** (on the real 8,523-row training set):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | 1069.38 | 792.03 | 0.579 |
| Linear Regression | 1069.38 | 792.04 | 0.579 |
| Tuned Ridge Regression (GridSearchCV) | 1069.58 | 791.79 | 0.579 |
| Random Forest | 1083.51 | 758.26 | 0.568 |
| Decision Tree | 1510.20 | 1030.40 | 0.161 |

> For reference, published leaderboard solutions on this dataset (with more extensive feature engineering) typically land around RMSE ~1090–1150, so these results are in a realistic, competitive range for the feature set used here.

---

## Testing & CI

Unit tests cover the core, side-effect-free pipeline functions (feature selection, cleaning, feature engineering, preprocessing, k-selection, clustering) in both projects.

```bash
pip install -r requirements-dev.txt
pytest tests/ -v          # run tests
ruff check src/ tests/ data/    # lint
black --check src/ tests/ data/ # formatting check
```

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs lint, formatting, unit tests, and a full smoke-test of both pipelines on every push/PR across Python 3.10–3.12, and uploads the generated plots/CSVs as build artifacts.

## Key Techniques Demonstrated

- Unsupervised learning: K-Means, elbow method, silhouette analysis, feature scaling
- Supervised learning: multi-model regression comparison, `sklearn` `Pipeline`/`ColumnTransformer` for leak-free preprocessing
- Feature engineering from raw/derived fields
- Hyperparameter tuning with `GridSearchCV`
- Data cleaning (label normalization, missing value imputation)
- Reproducible, CLI-driven scripts with saved artifacts (plots + CSVs)

## License

MIT — see [LICENSE](LICENSE).
