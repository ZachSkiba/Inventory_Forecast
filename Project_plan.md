# Retail Demand Forecasting System — Project Roadmap

## 🎯 What We Are Building
A time series forecasting system that predicts daily/weekly product demand using the M5 dataset (Walmart sales). This maps directly to what a supply chain or product data scientist at a Tier 1 company builds to optimize inventory, detect anomalies, and prevent stockouts.

**The Business Question:**
> *Given a product's price history, seasonality, and upcoming holidays, how many units will a specific store sell next week? Are sudden sales spikes genuine demand or anomalies?*

---

## 📅 Phase 1: Class Deliverables (EDA & Baselines)
*Goal: Establish a clean pipeline and prove statistical forecasting works on aggregate data.*

* **Notebook 1: EDA & Data Wrangling (`01_eda.ipynb`)**
    * Transform wide format (1,913 days) to long format using `pd.melt()`.
    * Merge calendar (events/holidays) and pricing data to compute revenue.
    * Analyze zero-inflated distributions (intermittent demand).
    * Perform seasonal decomposition on macro-level sales.
    * Select one representative product-store series for micro-level testing.
* **Notebook 2: Baselines & Statistical Models (`02_baselines_and_stats.ipynb`)**
    * Establish baselines: Naive forecast and 28-day Simple Moving Average (SMA).
    * Train SARIMA and Prophet on aggregate (macro) and single-item (micro) series.
    * Compare RMSE, MAE, and MAPE.

---

## 🚀 Phase 2: The ML Layer (XGBoost & Feature Engineering)
*Goal: Move from textbook statistics to production-grade machine learning.*

* **Notebook 3: Feature Engineering (`03_feature_engineering.ipynb`)**
    * Create temporal features: `day_of_week`, `is_weekend`, `month`.
    * Create event flags: `is_event_day`, state-specific `SNAP` days.
    * Create lag features: `lag_7`, `lag_28` (sales from 1 week / 4 weeks ago).
    * Create rolling metrics: `rolling_mean_7`, `rolling_mean_28`.
* **Notebook 4: XGBoost Forecasting (`04_xgboost_models.ipynb`)**
    * Train an XGBoost regressor using the engineered features.
    * Implement **Walk-Forward Cross-Validation** (critical for time-series validity).
    * Compare XGBoost performance against SARIMA/Prophet baselines.

---

## 🔍 Phase 3: Explainability & Anomaly Detection
*Goal: Show you can explain ML to stakeholders and detect business risks.*

* **Notebook 5: SHAP & Anomaly Detection (`05_explainability_anomalies.ipynb`)**
    * Generate **SHAP values** to explain which features drive the XGBoost predictions (e.g., proving a price drop and a holiday caused a sales spike).
    * Calculate residuals (Actuals - Predictions).
    * Flag anomalies using standard deviation thresholds (e.g., 3σ) or Isolation Forest to identify **Stockout Risks** (demand is unexpectedly 0) and **Viral Spikes**.

---

## 💻 Phase 4: Production Deployment (Streamlit)
*Goal: Build an interactive product for recruiters and hiring managers to click through.*

* **App Development (`app.py`)**
    * **Page 1: Inventory Control Panel:** Select a store and department to view a 28-day demand forecast.
    * **Page 2: Anomaly Alerts:** A feed of flagged "Stockout Risks" and "Demand Spikes".
    * **Page 3: Model Explainability:** Interactive SHAP charts explaining *why* the model is making its predictions to non-technical users.

---

## 📝 Phase 5: Portfolio Polish
*Goal: Package the project for Tier 1 recruiter screens.*

* **Clean GitHub Repo:** Well-structured folders (`data/`, `notebooks/`, `src/`).
* **Professional README:** Clear explanation of the business problem, methodology, data limitations, and how to run the app.
* **Resume Bullet:** > *"Developed an end-to-end retail demand forecasting system using the M5 dataset, processing 5+ years of daily sales across 30,000 product-store hierarchies. Engineered temporal, pricing, and event-driven features to train an XGBoost model, evaluated via walk-forward cross-validation. Deployed a Streamlit dashboard with SHAP explainability and statistical anomaly detection to surface automated inventory insights."*