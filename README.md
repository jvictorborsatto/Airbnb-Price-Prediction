# Airbnb Price Prediction

## Description

This project tackles the problem of predicting nightly rental prices for Airbnb listings scattered across the globe. Starting from a raw dataset of over 200,000 listings, the workflow moves through data cleaning, exploratory geographic analysis, and iterative model building with tree-based ensembles (Random Forest, Gradient Boosting, LightGBM). Along the way, the project investigates two competing strategies for handling the strong influence of location on price: training **one global model** using latitude/longitude as features (**Full Set**), versus training **separate models per region** — Europe, America, and Australia (**Region Sensitive**) — to capture market-specific pricing dynamics. Model performance is tracked with MAE, RMSE, and R², and hyperparameters are tuned step by step (n_estimators, max_depth, num_leaves, learning_rate) to squeeze out the best LightGBM configuration, validated with 5-fold cross-validation.

## Data Source

This project uses the dataset from the Kaggle competition **[Airbnb Rental Price Prediction (2025)](https://www.kaggle.com/competitions/rent-prediction-2025/overview)**, which challenges participants to predict rental prices for Airbnb listings. All credit for the original data goes to Kaggle and the competition organizers.

## Project Pipeline

The project is organized as three sequential notebooks:

| Order | Notebook | Purpose |
|---|---|---|
| 1 | `Data_exploration_clean_up.ipynb` | Load, clean, and segment the raw data by region |
| 2 | `Predictions_considering_latitude_and_longtude.ipynb` (→ **Full Set**) | Train a global model on the whole cleaned dataset, using lat/long as features |
| 3 | `Mult_Region_Model.ipynb` (→ **Region Sensitive**) | Train separate models per region (Europe, America, Australia) and compare against the global model |

---

## 1. Data Exploration & Clean-up

**Goal:** understand the raw `airbnb_train.csv` data, handle missing values, and prepare regional subsets for later modeling.

**Steps:**
- Loaded the raw dataset (**208,856 rows**) and inspected it with `.head()`, `.describe()`, and `.isna().sum()`.
- Dropped columns with more than **15% missing values**.
- Dropped remaining rows with any missing values, reducing the dataset from **208,757 → 180,695 rows**.
- Used `plotly.express` scatter plots of `longitude` vs `latitude` to visually inspect geographic coverage before and after dropping NaNs — this revealed that removing NaNs was cutting out parts of Western Australia, the Mediterranean, and the US.
- Filtered the data by lat/long bounding boxes into three regional subsets:
  - **Europe:** lat 20–80, long -50–50
  - **America:** lat -40–80, long -150–-50
  - **Australia:** lat -50–-10, long 100–150
- Saved the cleaned full dataset and the three regional subsets to `processed_data/` (`data.csv`, `data_europe.csv`, `data_america.csv`, `data_australia.csv`).

> 📊 The original lat/long scatter plots from this notebook were built with `plotly` (interactive charts) and weren't saved as static images inside the `.ipynb` file, so I can't re-embed the *exact* originals here. If you export them as PNG (`fig.write_image(...)`) or share the underlying CSV, I can drop the real ones in.

---

## 2. Full Set Model (Predictions considering latitude/longitude)

**Goal:** train a single global model on numeric features (including latitude/longitude) and find the best algorithm and hyperparameters.

**Steps:**
- Loaded the cleaned dataset, used a 20% sample to speed up initial model comparison.
- Trained and compared four baseline regressors on numeric features only:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 111.15 | 171.96 | 0.224 |
| Decision Tree | 101.89 | 166.62 | 0.272 |
| **Random Forest** | **72.12** | **112.69** | **0.667** |
| Gradient Boosting | 81.17 | 124.71 | 0.592 |
| LightGBM | 72.16 | 112.73 | 0.667 |

<img width="1760" height="720" alt="01_model_comparison" src="https://github.com/user-attachments/assets/4ecaaa94-8235-43dd-83e1-6213db3a83f6" />


- **LightGBM was chosen** to continue the investigation — it matched Random Forest's performance with much faster training.
- Ran feature importance analysis, which showed `id` and `host_id` had suspiciously high importance — a sign of **data leakage** from identifier columns rather than real predictive signal. These were dropped, while latitude/longitude were kept as meaningful geographic features.
- Re-trained LightGBM on the full dataset with the reduced, leakage-free feature set (`longitude`, `latitude`, `accommodates`, `host_total_listings_count`, `bathrooms`, `bedrooms`, `beds`, `number_of_reviews`, `number_of_reviews_ltm`, `number_of_reviews_l30d`) and tuned hyperparameters step by step:
  - **n_estimators:** tested 50/75/100/300/500/600 → best at **500–600** (diminishing returns above 500)
  - **max_depth:** tested 2/4/8/10/12 → little additional gain beyond 8
  - **num_leaves:** tested 32/42/64/86/128 → fixed at **64**
  - **learning_rate:** tested 0.01/0.03/0.05/0.1/0.2 → fixed at **0.1**

<img width="1760" height="1280" alt="02_hyperparameter_tuning" src="https://github.com/user-attachments/assets/acb69e3a-f314-4d09-abb3-5d654301738c" />


- Final tuned model (500 estimators, learning_rate 0.1, num_leaves 64, max_depth 8):

| Metric | Value |
|---|---|
| MAE | 67.02 |
| RMSE | 108.50 |
| R² | 0.694 |

- Validated with 5-fold cross-validation → **RMSE mean ≈ 108.30 (std ≈ 0.93)**, confirming the model is stable across folds.

<img width="1120" height="720" alt="05_cross_validation" src="https://github.com/user-attachments/assets/fcd58b01-a827-41db-b7aa-9c5e575004fb" />


- Feature importance (final model) confirmed **longitude and latitude as the two most important predictors**, followed by `host_total_listings_count` and `number_of_reviews`.

<img width="1280" height="800" alt="03_feature_importance_full_set" src="https://github.com/user-attachments/assets/7a027bba-fe0b-4ccb-833d-a37f5eb2b3b0" />


- Generated predictions on the held-out `airbnb_test.csv`.

---

## 3. Region Sensitive Model (Mult Region Model)

**Goal:** test whether training a separate model *per region* (instead of one global model) improves prediction accuracy, since price dynamics likely differ across markets.

**Steps:**
- Loaded the three regional subsets produced in step 1.
- For each region, built numeric features plus `room_type` and `property_type` (label-encoded), and dropped leakage-prone / redundant columns (`id`, `host_id`, `host_listings_count`, `availability_30/60/90/365`).
- Trained one **LightGBM** model per region (`objective='poisson'`, 500 estimators, learning_rate 0.2, num_leaves 64, max_depth 8).
- Feature importance for the Europe model again confirmed **longitude, latitude, and host_total_listings_count** as top predictors, with `number_of_reviews` and `property_type` also contributing meaningfully.

<img width="1280" height="880" alt="04_feature_importance_europe" src="https://github.com/user-attachments/assets/ef35ba5c-3f7c-41da-9597-a250b0bb7d9e" />


- Visualized predicted vs. actual prices for Australia with a `plotly` scatter plot annotated with RMSE/R² (interactive chart, same note as above — can be re-embedded as a static image if exported or if the raw CSV is shared).

---

## Key Findings

- **Tree-based ensemble models (Random Forest, LightGBM) massively outperform Linear Regression and single Decision Trees** for this price prediction task.
- **LightGBM was the best trade-off between accuracy and training speed**, and was used as the model of choice going forward.
- **Geographic location (latitude/longitude) is consistently the strongest predictor of price**, ahead of physical listing attributes like bedrooms/bathrooms.
- **Identifier columns (`id`, `host_id`) can leak information** and inflate feature importance without representing real signal — important to catch and remove during feature selection.
- The **region-sensitive approach** was explored to test whether localized models beat a single global model — see the note above regarding a variable-reuse bug in the America/Australia cells that should be fixed to get a clean, trustworthy comparison.

## Tech Stack

- **Data handling:** `pandas`, `numpy`
- **Visualization:** `plotly.express`
- **Modeling:** `scikit-learn` (Linear Regression, Decision Tree, Random Forest, Gradient Boosting), `lightgbm`
- **Validation:** `train_test_split`, `KFold` cross-validation

## Repository Structure

```
├── Data_exploration_clean_up.ipynb
├── Predictions_considering_latitude_and_longtude.ipynb   # Full Set model
├── Mult_Region_Model.ipynb                                # Region Sensitive model
├── processed_data/
│   ├── data.csv
│   ├── data_europe.csv
│   ├── data_america.csv
│   └── data_australia.csv
└── data_set/
    └── airbnb_test.csv
```

## How to Run

1. Run `Data_exploration_clean_up.ipynb` first to generate the cleaned/regional CSVs in `processed_data/`.
2. Run the **Full Set** notebook to train and evaluate the global model.
3. Run the **Region Sensitive** notebook to train and evaluate the per-region models.

