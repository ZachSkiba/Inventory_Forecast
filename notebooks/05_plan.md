# 05 — XGBoost Global Demand Model: M5 Walmart Demand Intelligence

## Notebook Plan

**File:** `05_xgboost_demand.ipynb`

**Goal:** Train a single global XGBoost model across all 30,490 product-store
series. Evaluate via walk-forward cross-validation. Demonstrate that incorporating
price, SNAP, and lag features closes the univariate signal ceiling identified in
notebooks 02 and 03.

**Inputs:**
- `../data/processed/features_train.parquet`
- `../data/processed/features_val.parquet`
- `../data/processed/feature_cols.pkl`

**Outputs:**
- `../data/processed/xgb_model.pkl`
- `../data/processed/xgb_predictions_val.parquet`
- `../data/processed/xgb_cv_results.csv`

---

## Evaluation Strategy

This notebook uses a three-tier evaluation structure. Each tier answers
a different question and uses the metric best suited to that question.
Using a single metric across all evaluation contexts is a common mistake
that either hides problems (RMSE on log scale obscures business impact)
or produces misleading numbers (MAPE on zero-inflated data is dominated
by structural gap rows, not forecast quality).

**Tier 1 — Optimization objective (Optuna and early stopping):**
RMSE on log1p scale. The target is already `log1p(units_sold)`, so this
measures error in the space the model is actually trained in. Handles
zeros naturally (`log1p(0) = 0`), treats all 30,490 series on a
comparable scale, is fast to compute, and matches XGBoost's internal
objective. This is the only metric used to make modeling decisions.

**Tier 2 — Fold-level model quality reporting:**
Three metrics computed on non-zero actual rows only (documented
explicitly in every output):
- RMSE on log1p scale — primary, consistent with optimization objective
- MAE on log1p scale — more robust to outliers than RMSE, reported alongside
- Bias (mean signed error on log1p scale) — catches systematic
  under/overprediction, which matters more for inventory decisions than
  random error of the same magnitude. A model that is consistently 10%
  low causes systematic stockouts; a model that is randomly ±20% does not.

Non-zero filter is applied because rows where `actual = 0` (structural gap
rows — product unavailable, not absent demand) contribute undefined or
astronomically large percentage errors. Excluding them is not hiding
information; it is being precise about what the model is being asked to do.
The null rate from gap-aware feature engineering (29.6% of rows) is reported
separately as a data quality metric, not conflated with forecast accuracy.

**Tier 3 — Signal ceiling and SARIMA/Prophet comparison:**
MAPE on non-zero actual rows, computed on a single restricted slice:
FOODS_3_163_CA_3 aggregated to monthly revenue. This is the only slice
that is directly comparable to the 22.22% SARIMA and 24.25% Prophet
benchmarks from notebooks 02 and 03, which were evaluated at the same
granularity. Every table and chart that shows this comparison explicitly
labels it as "representative series, monthly, non-zero rows" so the
restriction is visible. The global model quality (Tier 2) and this
comparison number (Tier 3) are never presented in the same table without
a clear label distinguishing them.

**Tier 4 — Quantile model evaluation:**
Pinball loss per quantile level, plus empirical coverage check. A 95th
percentile forecast is only useful if it actually contains the true value
roughly 95% of the time — this is called calibration and it is the correct
evaluation for probabilistic forecasts. Pinball loss is the standard metric
used in the M5 uncertainty track and in production probabilistic forecasting
systems. Point forecast MAPE is not the right metric for quantile models.

---

## Walk-Forward CV Structure

```
Feb 2011 ──────────────────────────────────────── Jan 2016

Fold 1:  [════════ TRAIN ════════][══ VAL ══]
          Feb 2011 → Jan 2013      Feb 2013 → Jan 2014
          Optuna tunes on this val window — exploratory

Fold 2:  [════════════ TRAIN ════════════][══ VAL ══]
          Feb 2011 → Jan 2014              Feb 2014 → Jan 2015
          Optuna tunes on this val window — primary tuning fold
          Rerun as many times as needed until satisfied
          Best params FROZEN here — never revisited after this

Fold 3:  [════════════════ TRAIN ════════════════][══ TEST ══]
          Feb 2011 → Jan 2015                      Feb 2015 → Jan 2016
          No tuning — Fold 2 frozen params applied directly
          Final benchmark vs SARIMA and Prophet
          Run exactly once. Never rerun after seeing results.
```

**Key rule:** Fold 3 test window is never used to make any modeling decision.
The moment it is, it stops being a test set and the reported performance
is no longer trustworthy.

---

## Tuning Philosophy

Sections 5 and 6 are the **tuning sandbox**. You are free to:
- Rerun Optuna with more trials
- Adjust search space ranges
- Add or remove parameters
- Iterate until satisfied

You are only ever looking at Fold 1 and Fold 2 val windows during this
process. Both are allowed to influence decisions — that is their purpose.

Once you move to Section 7, the rules change completely. One run, one result,
done.

---

## Section Plan

---

### Header & Goal Cell
- Notebook objective and role in the pipeline
- Inputs, outputs, evaluation strategy
- Demand proxy reminder
- Signal ceiling targets: beat SARIMA 22.22% and Prophet 24.25% on the
  representative series monthly slice (Tier 3 comparison)

---

### Section 1: Imports and Setup

- Imports: pandas, numpy, matplotlib, xgboost, optuna, sklearn, pickle
- Constants: file paths, random seed, fold date boundaries as explicit named constants
- Three evaluation functions defined once, used everywhere:

```python
def eval_log_scale(y_true_log, y_pred_log, label):
    """
    Tier 1 / Tier 2 metric — RMSE and MAE in log1p space.
    Optimization objective for Optuna and early stopping.
    Works on all rows including zeros (log1p(0) = 0).
    Also reports signed bias to catch systematic under/overprediction.
    """

def eval_unit_scale(y_true_units, y_pred_units, label):
    """
    Tier 3 metric — MAPE, MAE, bias in original unit space.
    Applied only on non-zero actual rows. Restriction documented on every call.
    Used exclusively for the signal ceiling comparison against SARIMA/Prophet
    on the representative series aggregated to monthly revenue.
    Never used as an optimization objective.
    """

def eval_quantiles(y_true, y_pred_q50, y_pred_q80, y_pred_q95, y_pred_q99, label):
    """
    Tier 4 metric — pinball loss per quantile level.
    Empirical coverage check: what % of actuals fall below each quantile band.
    Targets: q50 ~50%, q80 ~80%, q95 ~95%, q99 ~99%.
    A miscalibrated quantile model produces unsafe inventory decisions even
    if the point forecast is accurate — coverage is the primary output.
    Service level interpretation:
      q50 — median demand, central planning estimate
      q80 — moderate service level (cost-sensitive / low-velocity items)
      q95 — high service level (standard grocery / FOODS target)
      q99 — very high service level (staples / high stockout cost items)
    Wide bands on sparse HOBBIES series at q99 are not noise — they correctly
    communicate that hitting 99% service level on intermittent products
    requires holding substantial safety stock, which is why retailers
    do not target 99% uniformly across all categories.
    """
```

- `EARLY_STOPPING_ROUNDS` defined as a constant

---

### Section 2: Load Feature Matrix
- Load `features_train.parquet` and `features_val.parquet`
- Load `feature_cols.pkl`
- Confirm shapes, date ranges, series count, feature count
- Confirm val dates are strictly after train dates — final leakage check
- Report structural gap row rate (rows where all lag features are null) as
  a data quality metric separate from forecast accuracy

---

### Section 3: Walk-Forward CV Setup
- Define fold boundaries explicitly as a folds dictionary
- Print fold summary table: train rows, val rows, non-zero row counts,
  date ranges per fold
- Explain why walk-forward CV and not k-fold
- Explain the full evaluation strategy: Tier 1 is the optimization target,
  Tier 3 is reported at the end for the SARIMA/Prophet comparison
- Explain why the same metric is not used for both purposes

---

### Section 4: Default Model — Folds 1 and 2
- Train XGBoost with sensible default parameters on Fold 1 and Fold 2
- Use early stopping monitored on **RMSE (log scale)** on each fold's val
  window to find optimal n_estimators
- Time a single fit — this determines whether full-data tuning is feasible
- Report Tier 2 metrics (RMSE log, MAE log, bias) on each fold's val window
- Print results table: Fold 1 default log-RMSE, Fold 2 default log-RMSE
- This is the untuned floor Optuna must improve on

---

### Section 5: Hyperparameter Tuning — Fold 1 (Optuna)

**Objective function:** minimize RMSE on log1p scale on Fold 1 val window.
This is what the model is trained to optimize — using a different metric
for tuning (e.g. MAPE on unit scale) would create a mismatch between the
training objective and the tuning signal, leading to inconsistent behavior.

**Search space:**
- `max_depth`: 3–10
- `learning_rate`: 0.01–0.3 (log scale)
- `subsample`: 0.6–1.0
- `colsample_bytree`: 0.6–1.0
- `min_child_weight`: 1–50
- `reg_alpha`: 0–5
- `reg_lambda`: 0–5

`n_estimators` handled by early stopping — not a tuning parameter.
Including it in the search space while also using early stopping produces
inconsistent results because the two mechanisms interfere with each other.

- n_trials: 50
- Suppress Optuna trial-level logging — print only progress and best result
- Print best params and tuned log-RMSE
- Compare tuned vs default log-RMSE on Fold 1
- Purpose: exploratory — learning which parameters matter and what ranges work

---

### Section 6: Hyperparameter Tuning — Fold 2 (Optuna — Iterative)

Same Optuna setup as Section 5 but on Fold 2's larger training window.
Fold 2 is the primary tuning fold because its training window is closer
in size to Fold 3 — parameters that work well on 36 months of data may
not transfer well to parameters tuned on only 24 months.

- Warm start: initialize search space around Fold 1 best params
- n_trials: 50 to start — rerun with more trials if not converged
- **This section can and should be rerun freely**
- Each rerun adds trials and refines the search
- Print best params and tuned log-RMSE after each run
- Print comparison table: default vs tuned log-RMSE for both folds
- When satisfied with Fold 2 result → write best params as explicit constants
- **FREEZE PARAMS — do not modify after this point**

---

### Section 7: Final Model — Fold 3 (Test)

- Load frozen params from Section 6 constants
- Train on full Feb 2011 → Jan 2015 with frozen params
- Early stopping monitored on RMSE (log scale) on a held-out slice of
  the training data (last 60 days of training period) to set n_estimators.
  This slice is carved from training data only — not from the test window.
- Predict on Feb 2015 → Jan 2016

**Evaluation:**

*Global model quality (Tier 2):*
- RMSE on log scale, MAE on log scale, bias — across all non-zero rows
- Stratified by volume tier (high/medium/low velocity) to show where
  the model performs well and where it struggles
- MAPE by department and store — operational diagnostic, not the headline metric

*Signal ceiling comparison (Tier 3):*
- Filter predictions to FOODS_3_163_CA_3 only
- Aggregate daily predicted units × price to monthly revenue
- Compute MAPE vs actual monthly revenue on non-zero months
- Label explicitly: "representative series, monthly aggregation, non-zero months"
- Report month-by-month errors for all 12 test months
- **Signal ceiling test:** explicitly highlight Apr 2015, May 2015, Jan 2016
- Compare against SARIMA (22.22%) and Prophet (24.25%) on same slice
- This is the headline result of the project

**Run exactly once. Never rerun after seeing results.**

---

### Section 8: Quantile Models — Fold 3

Train four XGBoost models on Feb 2011 → Jan 2015 using frozen params
with quantile objective. Four quantiles are used rather than three because
real retail inventory decisions are made at specific service level targets
that vary by category — a single upper quantile cannot serve all use cases.

- `objective='reg:quantileerror'`, `quantile_alpha=0.50` — median demand,
  central planning estimate. Not the same as the point forecast (which
  minimizes squared error and estimates the conditional mean). On
  right-skewed zero-inflated demand, q50 will be below the point forecast
  — this is correct behavior, not an inconsistency.
- `objective='reg:quantileerror'`, `quantile_alpha=0.80` — moderate service
  level. Target for cost-sensitive or low-velocity items (HOBBIES category)
  where overstock cost is high and stockout consequence is low.
- `objective='reg:quantileerror'`, `quantile_alpha=0.95` — high service
  level. Standard target for grocery and FOODS items. Stocking to this
  level means covering demand on 95 out of every 100 days — roughly 18
  stockout days per year on a daily-selling product.
- `objective='reg:quantileerror'`, `quantile_alpha=0.99` — very high service
  level. Target for staples and high-velocity items where a stockout means
  a lost customer, not just a lost sale. Roughly 3.6 stockout days per year.
  Bands will be wide on sparse HOBBIES series — this is correct and
  intentional. It communicates that 99% service level on intermittent
  demand requires holding substantial safety stock.

**Why not q10 as a lower bound:**
q10 produces a lower bound that is operationally meaningless for inventory
decisions — no retailer stocks to a level that causes a stockout 90% of
the time. The lower bound of the uncertainty band (for visualization
purposes in the Streamlit app) is better represented by q50 itself, which
shows the level below which demand falls half the time. The inventory
decision is always about the upper tail, not the lower.

**Evaluation (Tier 4 — pinball loss + calibration):**

Pinball loss per quantile level:
```
pinball(q, y, y_hat) = q * max(y - y_hat, 0) + (1 - q) * max(y_hat - y, 0)
```
Lower is better. Upper quantile models (q95, q99) should overpredict more
than lower ones — pinball loss penalizes each direction asymmetrically and
in the right proportion for each quantile.

Empirical coverage check — the primary output:
- What % of actual values fall below the q50 forecast? Target: ~50%
- What % of actual values fall below the q80 forecast? Target: ~80%
- What % of actual values fall below the q95 forecast? Target: ~95%
- What % of actual values fall below the q99 forecast? Target: ~99%

A miscalibrated quantile model produces unsafe inventory decisions even
if the point forecast is accurate. A q95 model that only covers actuals
85% of the time is not a 95% service level tool — it is a mislabeled 85%
service level tool and will cause more stockouts than the retailer planned for.

Plot actual vs q50/q80/q95/q99 bands on representative series — visual
calibration check alongside the numerical coverage check. Report coverage
stratified by category (FOODS / HOUSEHOLD / HOBBIES) since calibration
may vary across demand profiles.

These four models feed directly into the Streamlit inventory scenario engine.
The Streamlit app labels them by service level interpretation, not by
quantile number, since service level is what a supply chain planner
understands and acts on.

---

### Section 9: Walk-Forward CV Results Summary

Master results table — Tier 2 metrics (log scale, all non-zero rows):

| Fold | Train Period | Test Period | Default log-RMSE | Tuned log-RMSE | Bias |
|---|---|---|---|---|---|
| 1 | Feb 2011 → Jan 2013 | Feb 2013 → Jan 2014 | x | x | x |
| 2 | Feb 2011 → Jan 2014 | Feb 2014 → Jan 2015 | x | x | x |
| 3 | Feb 2011 → Jan 2015 | Feb 2015 → Jan 2016 | — | x | x |
| **Avg (Folds 1+2)** | | | x | x | x |

Signal ceiling comparison table — Tier 3 metrics (representative series,
monthly, non-zero months only — directly comparable to SARIMA/Prophet):

| Model | Series | Granularity | MAPE | Notes |
|---|---|---|---|---|
| Naive | FOODS_3_163_CA_3 | Monthly revenue | 55.29% | |
| SMA(3) | FOODS_3_163_CA_3 | Monthly revenue | 45.80% | |
| SARIMA | FOODS_3_163_CA_3 | Monthly revenue | 22.22% | |
| Prophet | FOODS_3_163_CA_3 | Monthly revenue | 24.25% | |
| XGBoost | FOODS_3_163_CA_3 | Monthly revenue | x% | Daily preds aggregated to monthly |

These two tables are never merged — they use different metrics for
different purposes and combining them would be misleading.

---

### Section 10: Signal Ceiling Analysis

Side-by-side monthly error chart: SARIMA vs Prophet vs XGBoost on the
representative series. All three lines computed on the same slice
(FOODS_3_163_CA_3, monthly revenue, non-zero months).

- Highlight Apr 2015, May 2015, Jan 2016 explicitly with annotations
- Quantify the gap closed on each failure month vs SARIMA and Prophet
- Report which features most likely drove the improvement:
  - If `price_change_pct` / `price_drop` importance is high → confirms
    the Apr/May 2015 hypothesis (price-driven demand surge)
  - If `is_snap` importance is high → confirms the Jan 2016 hypothesis
    (SNAP distribution timing)
- Cross-reference with Section 11 global feature importance to validate

This is the headline result of the project. It should be the first thing
shown in the README and the Streamlit technical page.

---

### Section 11: Global Feature Importance

XGBoost gain-based importance — top 15 features bar chart.

Gain importance is preferred over split count for this analysis because
it measures how much each feature actually reduces the loss when it is
used for a split, not just how often it appears. A feature used frequently
for shallow splits (e.g. a categorical ID with many values) will have high
split count but may contribute little actual predictive value.

Expected ranking based on EDA:
- Lag features (`lag_1`, `lag_7`, `rolling_mean_7`) expected to dominate —
  SARIMA AR(1) p≈0.000 is the statistical basis for this
- `sell_price` and `price_change_pct` expected in top 10 —
  r=0.553 for price drops at product-store level (EDA Section 18)
- `is_snap` expected in top 15 — +10–32% FOODS uplift by state (EDA Section 12)
- If price and SNAP features do NOT appear in the top 15, the signal ceiling
  result in Section 10 needs a revised explanation

---

### Section 12: Error Analysis

All error analysis uses Tier 2 metrics (log-RMSE, bias) stratified by
dimension. MAPE is not used here — the zero-inflation problem is most
severe when slicing to sparse departments (HOBBIES_2) where most rows
are zero and MAPE becomes meaningless.

- **By department:** log-RMSE and bias per department. Identifies where
  the model struggles structurally (expected: HOBBIES_2 highest error,
  FOODS_3 lowest)
- **By store:** log-RMSE and bias per store. Key question: does TX_2
  improve over Prophet's 15.31% MAPE? Compute Tier 3 MAPE for TX_2
  separately (monthly aggregated revenue, non-zero months) for this
  specific comparison — label it clearly
- **Error distribution:** histogram of residuals in log space. Should be
  approximately symmetric around zero with no heavy systematic tail —
  a right-skewed residual distribution indicates the model is
  systematically underpredicting demand spikes
- **Hardest series:** identify the 10 series with highest log-RMSE.
  Are they concentrated in specific departments, stores, or volume tiers?
  This informs where LightGBM (notebook 06) may improve

---

### Section 13: Save Outputs

- Save `xgb_model.pkl` — the Fold 3 point forecast model
- Save `xgb_quantile_models.pkl` — dict of all four quantile models keyed
  by alpha (0.50, 0.80, 0.95, 0.99)
- Save `xgb_predictions_val.parquet` — Fold 3 predictions with columns:
  `id`, `date`, `units_sold`, `yhat`, `yhat_q50`, `yhat_q80`, `yhat_q95`, `yhat_q99`
  All values in original unit space (already back-transformed with expm1)
- Save `xgb_cv_results.csv` — all fold results for master comparison in
  notebook 06. Columns: `fold`, `metric`, `value`, `model`, `scope`
  where `scope` distinguishes global log-scale from representative-series
  unit-scale results so they are never accidentally compared
- Confirm file sizes and row counts

---

### Section 14: Summary

- What was built and how it was evaluated
- Why three evaluation tiers are necessary and what each measures
- Final CV results table (Tier 2)
- Signal ceiling comparison (Tier 3) — the headline result
- Quantile calibration summary (Tier 4)
- Key findings — did we close the signal ceiling? On which months?
  Which features drove it?
- Limitations:
  - Observed sales ≠ true demand — all metrics measure approximation
    of the demand proxy, not recovery of true latent demand
  - Gap-aware nulling handles structural zeros but not intermittent
    stockouts within active windows — these bias lag features slightly
    downward for high-velocity products
  - Global model treats all series equally during training — a
    production system would likely weight high-revenue series more
    heavily or train separate models by category
- What notebook 06 (LightGBM) does next and how results will be compared

---

## Key Rules for This Notebook

| Rule | Reason |
|---|---|
| Optuna objective is RMSE on log scale | Matches training objective — tuning on a different metric creates a training/tuning mismatch |
| Early stopping monitored on log-RMSE | Same reason — consistency between training signal and stopping criterion |
| MAPE never used as an optimization objective | Zero-inflation makes it unreliable on this dataset — 68% zero rows means MAPE is dominated by structural gaps, not forecast quality |
| Three evaluation tiers, never merged | Each tier answers a different question; merging them produces metrics that are neither interpretable nor comparable |
| Signal ceiling comparison restricted to representative series | The only way to compare against SARIMA/Prophet which were evaluated on a single monthly series — restriction labeled explicitly every time |
| Quantile models evaluated on pinball loss and coverage | MAPE does not measure whether a quantile forecast is calibrated — coverage does |
| Four quantiles not three | q50/q80/q95/q99 maps to real retail service level decisions; q10 has no operational meaning for inventory planning |
| q99 bands being wide on sparse series is correct | It communicates the true cost of 99% service level on intermittent demand — hiding this would produce unsafe inventory decisions |
| Fold 3 test window touched exactly once | Seeing the result and rerunning invalidates the test — this is the most important rule in the notebook |
| Fold 2 params written as explicit constants before Section 7 | Makes the freeze visible, auditable, and irreversible |
| n_estimators handled by early stopping only | Including it in Optuna search space while using early stopping creates inconsistent behavior |
| Bias reported alongside RMSE on every fold | Symmetric accuracy metrics hide directional problems — a model with 0 bias and high RMSE is better for inventory than one with low RMSE but systematic underprediction |
| All folds use identical FEATURE_COLS | Guarantees apples-to-apples comparison across folds |
| Quantile models use same frozen params as point model | Consistency across the forecast suite — different params would make the quantile bands inconsistent with the point forecast |
| Predictions saved in original unit space | Downstream Streamlit app and inventory math should not need to know about the log transform |