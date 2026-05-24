# Retail Demand Intelligence System — Project Roadmap

## 🎯 What We Are Building

A machine learning–driven demand intelligence system using the M5 Walmart dataset that models **product-level demand using observed sales as a proxy**.

True latent demand is not directly observable in this dataset (due to unknown stockouts or lost sales). Instead, this system approximates demand behavior using historical sales, pricing, and calendar effects — consistent with real-world retail forecasting pipelines when inventory data is unavailable.

---

### What Makes This Different

Unlike traditional forecasting projects, this system explicitly accounts for:

* demand distortions from pricing and promotions
* calendar and holiday effects
* intermittent and sparse SKU-level behavior with gap-aware feature engineering
* the gap between observed sales and true underlying demand
* multiple forecast horizons mapped to real inventory decisions

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
* All 30,490 product-store series are modeled — not just the 2,469 complete series
* Gap-aware lag logic handles the 430-day average zero streak confirmed in EDA
* Structural zeros (product unavailable) are distinguished from genuine zero demand where possible

---

## 📅 Phase 1: Data Engineering & Demand Understanding (EDA Layer)

*Goal: Understand demand structure and separate signal from noise.*

### Notebook 1: EDA & Data Wrangling (`01_eda.ipynb`) ✅

* Transform wide-format daily sales data into long format (`pd.melt()`)
* Merge calendar events (holidays, SNAP) and pricing data
* Construct revenue and demand proxies
* Analyze zero-inflation, sparse SKU-level behavior, aggregation effects
* Perform seasonal decomposition on aggregate and representative SKU-store series
* Quantify SNAP uplift by state, price elasticity at product-store level
* Confirm signal ceiling: univariate models fail on same 3 months (Apr/May 2015, Jan 2016)
* ADF stationarity tests and ACF/PACF parameter selection for SARIMA

---

## 📊 Phase 2: Baseline Forecasting (Statistical Benchmarks)

*Goal: Establish interpretable statistical baselines and confirm univariate signal ceiling.*

### Notebook 2: SARIMA Baselines (`02_baselines_and_stats.ipynb`) ✅

* Naive and SMA(3) baselines
* SARIMA on aggregate series: SARIMA(2,0,1)(0,1,1)[12] — MAPE 6.91%
* SARIMA on representative series: SARIMA(0,0,1)(0,1,1)[12] — MAPE 22.22%
* Signal ceiling identified: same 3 months fail regardless of model sophistication
* Prediction intervals produced analytically assuming Gaussian errors — miscalibrated
  on zero-inflated retail demand (25% violation rate on representative series vs
  expected 5%)

### Notebook 3: Prophet Baselines (`03_prophet.ipynb`, `03b_prophet_stores_conclusion.ipynb`) ✅

* Prophet on aggregate series — MAPE 5.02% (beats SARIMA via trend continuation)
* Prophet on representative series — MAPE 24.25% (statistical tie with SARIMA)
* Prophet on top 3 stores: CA_1 (2.66%), CA_3 (6.23%), TX_2 (15.31%)
* Signal ceiling confirmed: both model families fail on identical months
* TX_2 identified as clearest store-level case requiring external demand drivers
* Prediction intervals produced via Monte Carlo simulation — same miscalibration
  problem as SARIMA on zero-inflated series

---

## 🚀 Phase 3: Machine Learning Demand Modeling (Core System)

*Goal: Learn nonlinear demand behavior using feature-based models across all 30,490 series.*

---

### Notebook 4: Feature Engineering (`04_feature_engineering.ipynb`) ✅

#### Scope
* All 30,490 product-store series — not just complete series
* Gap-aware lag logic: lags spanning zero streaks > 28 consecutive days are nulled
* XGBoost and LightGBM handle nulls natively via surrogate splits

#### Features Built

| Group | Features | EDA Justification |
|---|---|---|
| Temporal (7) | `day_of_week`, `day_of_month`, `week_of_year`, `month_num`, `is_weekend`, `is_month_start`, `is_month_end` | Calendar cycles in seasonal decomposition |
| Event/SNAP (5) | `is_event`, `is_closed_holiday`, `days_to_closed_holiday`, `is_pre_closed_holiday`, `is_snap` | SNAP: +10–32% FOODS uplift by state. Christmas/Thanksgiving demand lives in lead days |
| Price (5) | `sell_price`, `price_change_pct`, `price_drop`, `price_increase`, `price_rel_28` | Asymmetric elasticity confirmed: r=0.553 drops vs r=0.18 increases at product-store level |
| Lag/Rolling (7) | `lag_1`, `lag_7`, `lag_14`, `lag_28`, `rolling_mean_7`, `rolling_mean_28`, `rolling_std_7` | SARIMA AR(1) p≈0.000 — recent demand is strongest predictor |
| Hierarchical (4) | `store_rolling_7`, `store_rolling_28`, `dept_rolling_7`, `dept_rolling_28` | Stores vary 3x in size; departments correlated 0.64–0.96 but not perfectly |
| Categoricals (6) | `store_id_enc`, `item_id_enc`, `dept_id_enc`, `cat_id_enc`, `state_id_enc`, `weekday_enc` | XGBoost needs integer-encoded IDs to learn per-store/dept demand baselines |
| **Total** | **34** | |

#### Deliberately Excluded

| Excluded | Reason |
|---|---|
| Department-level price features | r < 0.13 at department level — no signal (EDA Section 18) |
| Global SNAP flag | State-specific rates differ 3x — replaced by state-aware `is_snap` |
| Lags beyond 28 days | Matches forecast horizon — longer lags add noise, not signal |

#### Train / Validation Split

| Split | Dates | Rows | Purpose |
|---|---|---|---|
| Train | Feb 2011 → Jan 2015 | ~28.7M | Model learns from this |
| Val | Feb 2015 → Jan 2016 | ~11.1M | Evaluation — touched once at end of nb 05 |

#### Outputs
* `features_train.parquet` — training rows, all features, target
* `features_val.parquet` — validation rows (same schema)
* `feature_cols.pkl` — ordered feature list for notebook 05 and app layer
* `label_encoders.pkl` — for inverse-transform in Streamlit app

#### Key Engineering Decisions
* **Gap threshold = 28 days** — matches forecast horizon. Product absent for a full forecast period is structurally unavailable, not intermittently slow
* **29.6% of rows flagged as structural gaps** — consistent with EDA finding that 44.1% of series have zero streaks exceeding 365 days
* **`price_change_pct_raw` preserved** — 3,937 anomalous rows (M5 source data errors) kept for anomaly detection in notebook 07
* **`rolling_std_7`** feeds directly into safety stock formula in Streamlit app: `safety_stock = z × rolling_std_7 × √lead_time`

---

### Notebook 5: XGBoost Global Demand Model (`05_xgboost_demand.ipynb`)

#### Model
* **XGBoost** — single global model across all 30,490 series
* Point forecast model trained with `reg:squarederror` on `log1p(units_sold)` target
* Four separate quantile models trained with `reg:quantileerror` at service level
  quantiles that map directly to real inventory decisions — see Quantile Models below
* Replaces the miscalibrated parametric prediction intervals from SARIMA and Prophet
  with empirical quantile estimates that make no distributional assumption

#### Target
* `log1p(units_sold)` — back-transformed with `np.expm1` at inference
* Log transform reduces skewness from 17.18 to 1.93, preventing XGBoost from
  over-weighting the rare high-volume days at the expense of typical demand

#### Evaluation Strategy (Three Tiers)

A single metric across all evaluation contexts produces either hidden problems
or misleading numbers on zero-inflated retail data. Three tiers are used:

**Tier 1 — Optimization objective (Optuna and early stopping):**
RMSE on log1p scale. Matches the training objective, handles zeros naturally,
treats all 30,490 series on a comparable scale. The only metric used to make
modeling decisions.

**Tier 2 — Fold-level model quality reporting:**
RMSE on log1p scale, MAE on log1p scale, and signed bias — all computed on
non-zero actual rows. Bias catches systematic under/overprediction which matters
more for inventory decisions than random error of the same magnitude.

**Tier 3 — Signal ceiling and SARIMA/Prophet comparison:**
MAPE on non-zero rows, computed on a single restricted slice: FOODS_3_163_CA_3
aggregated to monthly revenue. The only slice directly comparable to the 22.22%
SARIMA and 24.25% Prophet benchmarks. Labeled explicitly as "representative
series, monthly, non-zero rows" every time it appears — never merged with
Tier 2 results.

**Tier 4 — Quantile model evaluation:**
Pinball loss per quantile level plus empirical coverage check. A q95 model is
only useful if actuals fall below it ~95% of the time. Coverage is the primary
output — a miscalibrated quantile model produces unsafe inventory decisions
even if the point forecast is accurate.

#### Walk-Forward Cross-Validation

Three expanding windows simulating production deployment:

| Fold | Train | Val Window | Role |
|---|---|---|---|
| 1 | Feb 2011 → Jan 2013 | Feb 2013 → Jan 2014 | Exploratory tuning |
| 2 | Feb 2011 → Jan 2014 | Feb 2014 → Jan 2015 | Primary tuning — params frozen here |
| 3 | Feb 2011 → Jan 2015 | Feb 2015 → Jan 2016 | Final benchmark — run once |

Fold 3 is the primary comparison benchmark against SARIMA and Prophet.
Optuna tunes on Folds 1 and 2 only. Fold 3 test window is never used to
make any modeling decision — the moment it is, the reported performance
is no longer trustworthy.

#### Hyperparameter Tuning
* Bayesian optimization (Optuna) — objective is RMSE on log1p scale
* Tuned on Fold 1 (exploratory), then Fold 2 (primary — params frozen after this)
* Search space: `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`,
  `min_child_weight`, `reg_alpha`, `reg_lambda`
* `n_estimators` handled by early stopping — never included in search space

#### Signal Ceiling Test
Evaluate on Apr 2015, May 2015, Jan 2016 — the three months where SARIMA and
Prophet both failed (>40% error on representative series). If XGBoost's price
and SNAP features close these gaps, it confirms the signal ceiling diagnosis
from notebooks 02 and 03. This is the headline result of the project.

#### Quantile Models

Four quantile models trained on the same Fold 3 training data with frozen params.
Quantiles chosen to map directly to real retail service level decisions:

| Quantile | Service Level Interpretation | Retail Use Case |
|---|---|---|
| q50 | Median demand — central planning estimate | Baseline order quantity; expected demand |
| q80 | 80% service level | Cost-sensitive / low-velocity items (HOBBIES). Overstock cost high. |
| q95 | 95% service level | Standard grocery / FOODS target. ~18 stockout days/year. |
| q99 | 99% service level | Staples / high-velocity items. ~3.6 stockout days/year. Stockout = lost customer. |

Wide bands at q99 on sparse HOBBIES series are correct and intentional —
they communicate the true inventory cost of a 99% service level on intermittent
demand. This is useful information, not noise.

These replace the "conservative/expected/aggressive" framing from earlier
versions. The Streamlit app labels bands by service level percentage, not
by abstract scenario name, because service level is what supply chain
planners act on.

Note on aggregation: point forecast (`yhat`) can be summed across series
to produce store-level or category-level demand estimates. Quantile bands
cannot be summed directly — summing q95 predictions across series produces
something higher than the true q95 of the aggregate due to the diversification
effect. Quantile bands are most accurate and most operationally meaningful
at the individual `item_id × store_id` level where inventory decisions are made.

#### Outputs
* `xgb_model.pkl` — Fold 3 point forecast model
* `xgb_quantile_models.pkl` — dict of four quantile models keyed by alpha
  (0.50, 0.80, 0.95, 0.99)
* `xgb_predictions_val.parquet` — Fold 3 predictions with columns:
  `id`, `date`, `units_sold`, `yhat`, `yhat_q50`, `yhat_q80`, `yhat_q95`, `yhat_q99`
  All values in original unit space (back-transformed with expm1)
* `xgb_cv_results.csv` — walk-forward results per fold, metric, and scope

---

### Notebook 6: LightGBM Global Demand Model (`06_lightgbm_demand.ipynb`)

#### Model
* **LightGBM** — same feature matrix as XGBoost, independent benchmark
* Faster on sparse data, often outperforms XGBoost on zero-inflated distributions
* Same four quantile models (q50, q80, q95, q99)
* Same walk-forward CV folds and three-tier evaluation — results directly comparable

#### Why a Separate Notebook
* Clean single-purpose notebooks are easier to navigate
* Keeps XGBoost results uncontaminated — each model is a standalone experiment
* Direct apples-to-apples comparison via identical folds, features, and metrics

#### Master Comparison Table (target — populated after both notebooks run)

Primary comparison metric is log-RMSE (Tier 2) across all folds.
SARIMA and Prophet Tier 3 MAPE shown separately for the signal ceiling comparison —
these are not the same metric and are never presented in the same column.

| Fold | Train Period | Val Period | Naive log-RMSE | XGBoost log-RMSE | LightGBM log-RMSE |
|---|---|---|---|---|---|
| 1 | Feb 2011 → Jan 2013 | Feb 2013 → Jan 2014 | x | x | x |
| 2 | Feb 2011 → Jan 2014 | Feb 2014 → Jan 2015 | x | x | x |
| 3 | Feb 2011 → Jan 2015 | Feb 2015 → Jan 2016 | x | x | x |
| **Avg** | | | x | x | x |

Signal ceiling comparison (Tier 3 — representative series, monthly, non-zero months):

| Model | MAPE |
|---|---|
| Naive | 55.29% |
| SMA(3) | 45.80% |
| SARIMA | 22.22% |
| Prophet | 24.25% |
| XGBoost | x% |
| LightGBM | x% |

#### Winner Selection
* Lower average log-RMSE across 3 folds wins (Tier 2 — primary criterion)
* Tie-broken by Tier 3 MAPE on representative series if log-RMSE is within 0.01
* Winner used for Pages 1 and 3 in Streamlit (live inventory decisions)
* Both models shown on Pages 2 and 5 (backtesting and technical comparison)
* Winner documented in README with explicit reasoning

#### Outputs
* `lgbm_model.pkl` — trained model
* `lgbm_quantile_models.pkl` — dict of four quantile models keyed by alpha
  (0.50, 0.80, 0.95, 0.99)
* `lgbm_predictions_val.parquet` — val period predictions (same schema as XGBoost)
* `lgbm_cv_results.csv` — walk-forward results per fold

---

## 🔍 Phase 4: Explainability & Demand Diagnostics

*Goal: Understand model behavior and detect risk patterns.*

---

### Notebook 7: SHAP & Anomaly Detection (`07_explainability.ipynb`)

#### Explainability (SHAP)
* Global feature importance — which features drive demand most across all SKUs
* Per-SKU SHAP waterfall plots — what drove this specific product's forecast
* Confirm price and SNAP features close the signal ceiling months identified in baselines
* Example output: *"Demand increase driven by SNAP event + price drop of 12% vs 28-day baseline"*

#### Anomaly Detection
* Residual = `actual - predicted`
* Flag demand spikes (actual >> predicted) and suppressed demand (actual << predicted)
* Z-score thresholds on residuals per series
* Isolation Forest on residual distribution
* Distinguish genuine demand anomalies from supply-side censoring events (product returning from structural gap)
* `price_change_pct_raw` anomalies (3,937 rows flagged in feature engineering) fed as candidates

---

## 💻 Phase 5: Deployment Layer (Decision System)

*Goal: Convert forecasts into actionable decisions.*

---

### Streamlit App (`app.py`)

Must load fast, look clean, tell a story in under 60 seconds. Performance is
non-negotiable — pre-compute all forecasts and store as parquet. No model
inference at runtime.

**Model used:** Winner of XGBoost vs LightGBM comparison (lower average
log-RMSE across 3 CV folds) powers Pages 1 and 3. Both models shown on
Pages 2 and 5 for comparison.

**Aggregation note:** The app supports filtering to any individual
`item_id × store_id` series or any higher-level aggregation (store, department,
category, state). Point forecasts aggregate correctly by summing. Quantile
bands at aggregated levels are labeled as indicative only — summing q95
predictions across series overstates the true aggregate q95 due to demand
diversification. Quantile bands are most reliable and most operationally
meaningful at the individual series level.

---

### Page 1: Inventory Decision Dashboard

**Inputs:**
* Store / SKU selection (any individual `item_id × store_id` combination)
* Current inventory level
* Supplier lead time (days)
* Target service level (%) — maps to q50 / q80 / q95 / q99 quantile model

**Outputs:**
* 28-day demand forecast with q50 / q80 / q95 / q99 service level bands
* Inventory trajectory simulation under all four service level scenarios
* Projected stockout date per scenario
* Stockout probability
* Recommended reorder quantity
* Recommended reorder timing
* Safety stock suggestion: `z × rolling_std_7 × √lead_time`
* Active/gap state flag — communicates forecast confidence to user

Service level bands labeled by percentage (50% / 80% / 95% / 99%), not by
abstract scenario name. Wide bands on sparse series are shown as-is —
they correctly communicate the inventory cost of high service levels on
intermittent demand.

> This transforms forecasting into a **decision engine**.

---

### Page 2: Simulation & Backtesting

* Walk-forward CV results table — log-RMSE and bias per fold per model
* Historical inventory simulation across val period (Feb 2015 → Jan 2016)
* Inventory depletion modeling
* Metrics: stockout rate, average inventory, demand coverage
* Four scenarios: q50 / q80 / q95 / q99 service level bands
* Signal ceiling visualization: error by month for SARIMA vs Prophet vs
  XGBoost on representative series (Tier 3 comparison — labeled explicitly)

---

### Page 3: Risk Monitoring

* Stockout risk alerts across all SKUs
* Demand spike detection (residual z-score > threshold)
* Overstock warnings
* Prediction drift tracking — log-RMSE by week over val period

---

### Page 4: Explainability Engine

* Global SHAP feature importance bar chart
* Per-SKU SHAP waterfall plot
* Natural language explanation of top 3 demand drivers for selected SKU
* Price elasticity visualization for selected product-store

---

### Page 5: Technical Deep Dive

* Walk-forward CV results table (Naive vs SARIMA vs Prophet vs XGBoost vs LightGBM)
  — two separate tables: Tier 2 log-RMSE for ML models, Tier 3 MAPE for
  signal ceiling comparison. Never merged.
* Signal ceiling chart: SARIMA vs Prophet vs XGBoost error by month on
  representative series
* Error distributions per model
* Global feature importance
* log-RMSE and bias by department and store
* Quantile calibration summary: empirical coverage vs target per quantile level

---

## ⚙️ Production System Design

*Goal: Show understanding of real-world ML systems.*

---

### Data Pipeline
* Daily ingestion of sales, pricing, calendar data
* Feature generation (lags, rolling stats, events)
* Gap detection per series on ingestion

### Model Pipeline
* Global XGBoost + LightGBM models
* Retraining weekly/monthly via walk-forward expanding window
* Quantile models (q50, q80, q95, q99) retrained alongside point forecast model

### Inference
* Batch forecasting at 7, 28, and 90-day horizons
* Stored predictions as parquet for downstream use and Streamlit performance

### Decision Layer
* Converts forecasts → inventory actions per service level target
* Estimates stockout risk under lead time uncertainty
* Safety stock recommendations per SKU based on target service level

### Monitoring
* log-RMSE drift tracking by week
* Bias monitoring — flags if model becomes systematically directional
* Anomaly detection on residuals
* Quantile calibration monitoring — flags if empirical coverage drifts
  away from target (e.g. q95 model only covering 85% of actuals)
* Gap state monitoring — flag series transitioning in/out of structural gaps

---

## 📝 Phase 6: Portfolio & Communication

---

### GitHub Structure

```
retail-demand-intelligence/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines_and_stats.ipynb
│   ├── 03_prophet.ipynb
│   ├── 03b_prophet_stores_conclusion.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_xgboost_demand.ipynb
│   ├── 06_lightgbm_demand.ipynb
│   └── 07_explainability.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── decisions/
├── app.py
├── requirements.txt
└── README.md
```

---

### README Must-Haves

* System architecture diagram
* Walk-forward CV results table with all models (Tier 2 log-RMSE)
* Signal ceiling visualization — the three failure months (Tier 3 MAPE on
  representative series — labeled as such)
* Three bullets on what makes this different from a standard forecasting project
* Screenshot or GIF of Streamlit app
* Clear distinction: **sales vs demand vs decisions**
* Note on quantile bands: service level interpretation, not prediction intervals

---

### Resume Bullet

Built an end-to-end retail demand intelligence system using the M5 Walmart dataset,
modeling product-level demand across 30,490 store-SKU combinations using observed
sales as a proxy. Engineered gap-aware lag, pricing, and calendar features across
all series to train global XGBoost and LightGBM models with quantile regression
(q50/q80/q95/q99 service level bands) under walk-forward cross-validation,
benchmarking against SARIMA and Prophet baselines. Identified and closed the
univariate signal ceiling — the three months where statistical models fail —
by incorporating price elasticity and SNAP distribution features. Developed a
decision-support system converting probabilistic demand forecasts into inventory
actions including stockout risk estimation, safety stock recommendations, and
reorder timing at user-specified service levels. Deployed an interactive Streamlit
application with SHAP-based explainability and multi-scenario inventory simulation.

---

## 🗂️ Key Numbers to Know

| Metric | Value |
|---|---|
| Total product-store series | 30,490 |
| Complete series (all 64 months) | 2,469 (8.1%) |
| Rows in feature matrix (train) | ~28.7M |
| Rows in feature matrix (val) | ~11.1M |
| Structural gap rows (29.6%) | ~17.3M |
| SARIMA aggregate MAPE | 6.91% |
| Prophet aggregate MAPE | 5.02% |
| SARIMA representative MAPE | 22.22% |
| Prophet representative MAPE | 24.25% |
| Signal ceiling months | Apr 2015, May 2015, Jan 2016 |
| Gap threshold | 28 consecutive zero-sales days |
| Forecast horizons | 7, 28, 90 days |
| Train period | Feb 2011 → Jan 2015 |
| Val period | Feb 2015 → Jan 2016 |
| Quantile service levels | q50 / q80 / q95 / q99 |
| Primary CV metric | log-RMSE (Tier 2) |
| Signal ceiling comparison metric | MAPE on representative series, monthly (Tier 3) |
| Quantile evaluation metric | Pinball loss + empirical coverage (Tier 4) |