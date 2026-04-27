# Retail Demand Intelligence System — Project Roadmap

## 🎯 What We Are Building

A machine learning–driven demand intelligence system using the M5 Walmart dataset that models **product-level demand using observed sales as a proxy**.

True latent demand is not directly observable in this dataset (due to unknown stockouts or lost sales). Instead, this system approximates demand behavior using historical sales, pricing, and calendar effects — consistent with real-world retail forecasting pipelines when inventory data is unavailable.

---

### What Makes This Different

Unlike traditional forecasting projects, this system explicitly accounts for:

* demand distortions from pricing and promotions
* calendar and holiday effects
* intermittent and sparse SKU-level behavior
* the gap between observed sales and true underlying demand

This reflects how modern supply chain and retail analytics teams design **decision-support systems** for inventory planning, risk monitoring, and demand forecasting.

---

## 🧠 Core Business Question

> *Given historical sales, pricing, calendar events, and lagged demand signals, how will demand evolve for a product-store combination, and how should inventory decisions adapt to avoid stockouts and overstock?*

This project goes beyond prediction:

> It converts forecasts into **actionable inventory decisions under uncertainty**.

---

## 🧩 System Scope & Assumptions

* Demand is approximated using observed sales (no inventory or stockout data available)
* The model does not directly observe lost sales or true latent demand
* Forecasts are used to simulate inventory behavior, not optimize it exactly
* The system is a **decision-support tool**, not a full inventory optimization engine

---

## 📅 Phase 1: Data Engineering & Demand Understanding (EDA Layer)

*Goal: Understand demand structure and separate signal from noise.*

### Notebook 1: EDA & Data Wrangling (`01_eda.ipynb`)

* Transform wide-format daily sales data into long format (`pd.melt()`)
* Merge calendar events (holidays, SNAP) and pricing data
* Construct revenue and demand proxies
* Analyze:

  * zero-inflation (intermittent demand)
  * sparse SKU-level behavior
  * aggregation effects (SKU vs store vs state)
* Perform seasonal decomposition on:

  * aggregate demand
  * representative SKU-store series
* Introduce **observed sales vs demand approximation**

---

## 📊 Phase 2: Baseline Forecasting (Statistical Benchmarks)

*Goal: Establish interpretable statistical baselines.*

### Notebook 2: Baselines & Statistical Models (`02_baselines.ipynb`)

* Naive forecast (last value)
* 28-day moving average
* SARIMA models:

  * aggregate series
  * representative SKU
* Prophet models:

  * trend + seasonality decomposition
  * holiday effects

### Key Insight

These models are not final solutions — they represent:

> **classical statistical approximations of demand structure**

They serve as benchmarks for evaluating machine learning models.

---

## 🚀 Phase 3: Machine Learning Demand Modeling (Core System)

*Goal: Learn nonlinear demand behavior using feature-based models.*

---

### Notebook 3: Feature Engineering (`03_feature_engineering.ipynb`)

#### Temporal Features

* `day_of_week`, `month`, `is_weekend`

#### Lag Features

* `lag_1`, `lag_7`, `lag_14`, `lag_28`
* `rolling_mean_7`, `rolling_mean_28`
* `rolling_std_7`

#### Pricing Features

* price changes
* lagged price ratios
* price volatility

#### Event Features

* holiday indicators (SNAP, Christmas, etc.)
* event windows (pre/post effects)

#### Hierarchical Features (Important)

* store-level rolling demand
* department-level aggregates
* category-level signals

---

### Notebook 4: XGBoost Global Demand Model (`04_xgboost_demand.ipynb`)

* Train a **single global model across all SKUs**
* Target: `log1p(sales)` for stability
* Use **walk-forward validation**
* Compare against:

  * SARIMA
  * Prophet

### Objective

> Capture nonlinear demand patterns that statistical models miss:

* price sensitivity
* lag dependencies
* event-driven spikes

---

## 🔍 Phase 4: Explainability & Demand Diagnostics

*Goal: Understand model behavior and detect risk patterns.*

---

### Notebook 5: SHAP & Anomaly Detection (`05_explainability.ipynb`)

#### Explainability (SHAP)

* Compute SHAP values
* Identify key drivers:

  * price changes
  * lag effects
  * holiday impact

Example:

> “Demand increase driven by SNAP event + recent upward trend”

---

#### Anomaly Detection

Residual:

```
residual = actual - predicted
```

Detect:

* demand spikes (actual >> predicted)
* suppressed demand (actual << predicted)
* unusual behavior patterns

Methods:

* z-score thresholds
* Isolation Forest (optional)

---

## 💻 Phase 5: Deployment Layer (Decision System)

*Goal: Convert forecasts into actionable decisions.*

---

### Streamlit App (`app.py`)

---

### Page 1: Inventory Decision Dashboard

**Inputs:**

* Store / SKU selection
* Current inventory
* Lead time
* Service level

**Outputs:**

* 28-day demand forecast
* Confidence intervals
* Inventory trajectory simulation
* Projected stockout date
* Stockout probability
* Recommended reorder quantity
* Recommended reorder timing
* Safety stock suggestion

> This transforms forecasting into a **decision engine**.

---

### Page 2: Simulation & Backtesting

* Historical simulation across test period
* Inventory depletion modeling
* Metrics:

  * stockout rate
  * average inventory
  * demand coverage

Scenarios:

* conservative
* expected
* aggressive

---

### Page 3: Risk Monitoring

* stockout risk alerts
* demand spikes
* overstock warnings
* prediction drift tracking

---

### Page 4: Explainability Engine

* SHAP visualizations
* feature contributions
* natural language explanations

---

### Page 5: Technical Deep Dive

* model comparison (SARIMA vs Prophet vs XGBoost)
* walk-forward validation plots
* error distributions
* global feature importance
* performance across SKUs

---

## ⚙️ Production System Design

*Goal: Show understanding of real-world ML systems.*

---

### Data Pipeline

* daily ingestion of sales, pricing, calendar data
* feature generation (lags, rolling stats, events)

### Model Pipeline

* global XGBoost model
* retraining weekly/monthly

### Inference

* batch forecasting (28-day horizon)
* stored predictions for downstream use

### Decision Layer

* converts forecasts → inventory actions
* estimates stockout risk

### Monitoring

* error tracking (MAPE drift)
* anomaly detection
* model performance over time

---

## 📝 Phase 6: Portfolio & Communication

---

### GitHub Structure

* modular `src/` pipeline
* separate layers:

  * data
  * features
  * models
  * decisions
* reproducible experiments

---

### README Highlights

* clear distinction: **sales vs demand vs decisions**
* business framing: *inventory decision support*
* model comparison
* walk-forward validation explanation
* simulation + business impact
* production system overview

---

### Resume Bullet

Built an end-to-end retail demand intelligence system using the M5 dataset, modeling product-level demand across 30,000+ store-SKU combinations using observed sales as a proxy. Engineered lag-based, pricing, and calendar features to train a global XGBoost model under walk-forward validation, benchmarking against SARIMA and Prophet baselines. Developed a decision-support system that converts demand forecasts into inventory actions, including stockout risk estimation, reorder recommendations, and anomaly detection. Deployed an interactive Streamlit application with SHAP-based explainability for real-time decision support.
