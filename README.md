# Retail Demand Forecasting System — M5 Walmart Sales

End-to-end time series forecasting system predicting retail product demand using the M5 Forecasting dataset (Walmart, 2020). Models daily/weekly sales at both the aggregate platform level and the individual product-store series level. Downstream output includes automated anomaly detection for stockout risk and viral demand spikes.

## Business Question

> Inventory decisions made without reliable demand forecasts result in stockouts that permanently lose customers and overstock that destroys margin. This project builds a system that predicts product-level demand 28 days out across 10 stores, quantifies the revenue impact of holidays, SNAP benefit days, and price changes, and automatically flags anomalous demand events before they become supply chain failures.

---

## Dataset

**M5 Forecasting — Accuracy** ([Kaggle, 2020](https://www.kaggle.com/competitions/m5-forecasting-accuracy))

| File | Description |
|---|---|
| `sales_train_validation.csv` | 30,490 product-store series × 1,913 days (wide format) |
| `calendar.csv` | Day IDs mapped to real dates, holidays, SNAP benefit flags |
| `sell_prices.csv` | Weekly prices per product per store |

- **Date range:** Jan 29 2011 – Jun 19 2016 (~5.25 years, 64 months)
- **Scale:** 3,049 products, 10 stores, 3 states (CA, TX, WI)
- **Hierarchy:** 3 categories → 7 departments → 30,490 product-store series

> Download all three files from Kaggle and place them in `data/raw/`.

---

## Project Structure

```
├── data/
│   ├── raw/
│   │   ├── sales_train_validation.csv
│   │   ├── calendar.csv
│   │   └── sell_prices.csv
│   └── processed/
│       ├── monthly_aggregate.csv
│       ├── monthly_series_FOODS_3_163_CA_3_validation.csv
│       └── complete_series_ids.csv
├── notebooks/
│   ├── 01_eda_w.ipynb
│   ├── 02_baselines_and_stats_w.ipynb
│   ├── 03_feature_engineering_w.ipynb
│   ├── 04_xgboost_models_w.ipynb
│   └── 05_explainability_anomalies_w.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Notebooks

### `01_eda_w.ipynb` — Exploratory Data Analysis ✅
- Reshape wide format (1,913 day columns) to long format via `pd.melt()` — 58M rows
- Join calendar (events, holidays, SNAP flags) and compute revenue via memory-efficient aggregated price joins
- Analyze zero-inflation: 68.2% of product-store-days have zero sales
- Quantify SNAP uplift by state (WI FOODS: +32.5%), holiday effects (Labor Day: +19.6%, Christmas: -100%)
- Seasonal decomposition and price volatility analysis across all 7 departments
- Select representative series: **FOODS_3_163_CA_3_validation** (median-revenue complete series)
- Save processed files for downstream modeling notebooks

**Key findings:**
- FOODS dominates at 58% of total revenue; FOODS_3 alone is 37.8%
- CA_3 is the highest-revenue store (17.1%) — nearly 3× the smallest store
- Strong upward trend dominates over seasonality in the aggregate series
- Only 2,469 of 30,490 series (8.1%) have complete 64-month histories

### `02_baselines_and_stats_w.ipynb` — Baselines & Statistical Models 🔜
- ADF stationarity test + ACF/PACF analysis to determine SARIMA order
- Naive and 28-day SMA baselines
- SARIMA on aggregate monthly revenue and representative individual series
- Prophet on both series
- Evaluation: RMSE, MAE, MAPE — 48-month train / 15-month test split

### `03_feature_engineering_w.ipynb` — Feature Engineering 🔜
- Temporal features: `day_of_week`, `is_weekend`, `month`, `year`
- Event flags: `is_event_day`, state-specific SNAP flags (`snap_CA`, `snap_TX`, `snap_WI`)
- Holiday lead features: `days_until_thanksgiving`, `days_until_christmas`
- Lag features: `lag_7`, `lag_28`
- Rolling features: `rolling_mean_7`, `rolling_mean_28`
- Price features: `sell_price`, `price_lag_1wk`, `price_change_pct`

### `04_xgboost_models_w.ipynb` — XGBoost Forecasting 🔜
- Train XGBoost regressor on engineered features
- Walk-forward cross-validation (no data leakage)
- Compare against SARIMA/Prophet baselines on RMSE, MAE, MAPE

### `05_explainability_anomalies_w.ipynb` — SHAP & Anomaly Detection 🔜
- SHAP values to explain feature contributions to XGBoost predictions
- Residual analysis (Actuals − Predictions)
- Anomaly flagging via 3σ threshold and Isolation Forest
- Identify stockout risks and viral demand spikes

---

## Streamlit App (`app.py`) 🔜

| Page | Description |
|---|---|
| Inventory Control Panel | Select a store and department to view a 28-day demand forecast |
| Anomaly Alerts | Feed of flagged stockout risks and demand spikes |
| Model Explainability | Interactive SHAP charts for non-technical stakeholders |

---

## Hardcoded Parameters (consistent across all notebooks)

| Parameter | Value |
|---|---|
| Representative series ID | `FOODS_3_163_CA_3_validation` |
| Seasonal period | `12` |
| Train / test split | 48 months train, 15 months test |
| SARIMA/Prophet forecast horizon | 12 months |
| XGBoost forecast horizon | 28 days |

---

## Resume Bullet

> *Developed an end-to-end retail demand forecasting system using the M5 dataset, processing 5+ years of daily sales across 30,000 product-store hierarchies. Engineered temporal, pricing, and event-driven features to train an XGBoost model, evaluated via walk-forward cross-validation. Deployed a Streamlit dashboard with SHAP explainability and statistical anomaly detection to surface automated inventory insights.*