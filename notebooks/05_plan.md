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

## Walk-Forward CV Structure

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


**Key rule:** Fold 3 test window is never used to make any modeling decision.
The moment it is, it stops being a test set and your reported performance
is no longer trustworthy.

---

## Tuning Philosophy

Sections 5 and 6 are your **tuning sandbox**. You are free to:
- Rerun Optuna with more trials
- Adjust search space ranges
- Add or remove parameters
- Iterate until satisfied

You are only ever looking at Fold 1 and Fold 2 val windows during this
process. Both are allowed to influence your decisions — that is their purpose.

Once you move to Section 7, the rules change completely. One run, one result,
done.

---

## Section Plan

---

### Header & Goal Cell
- Notebook objective and role in the pipeline
- Inputs, outputs, evaluation strategy
- Demand proxy reminder
- Signal ceiling targets: beat SARIMA 22.22% and Prophet 24.25% on Fold 3

---

### Section 1: Imports and Setup
- Imports: pandas, numpy, matplotlib, xgboost, optuna, sklearn, pickle
- Constants: file paths, random seed, fold date boundaries as explicit named constants
- `evaluate()` function — RMSE, MAE, MAPE — defined once, used everywhere
- Early stopping rounds defined as a constant

---

### Section 2: Load Feature Matrix
- Load `features_train.parquet` and `features_val.parquet`
- Load `feature_cols.pkl`
- Confirm shapes, date ranges, series count, feature count
- Confirm val dates are strictly after train dates — final leakage check

---

### Section 3: Walk-Forward CV Setup
- Define fold boundaries explicitly as a folds dictionary
- Print fold summary table: train rows, val rows, date ranges per fold
- Explain why walk-forward CV and not k-fold
- Explain the full tuning strategy: iterative on Folds 1 and 2, frozen on Fold 3

---

### Section 4: Default Model — Folds 1 and 2
- Train XGBoost with sensible default parameters on Fold 1 and Fold 2
- Use early stopping on each fold's val window to find optimal n_estimators
- Time a single fit — this determines whether full-data tuning is feasible
- Evaluate MAPE on each fold's val window after back-transforming with expm1
- Print results table: Fold 1 default MAPE, Fold 2 default MAPE
- This is the untuned floor Optuna must improve on

---

### Section 5: Hyperparameter Tuning — Fold 1 (Optuna)
- Optuna objective: minimize MAPE on Fold 1 val window
- Search space:
  - `max_depth`: 3–10
  - `learning_rate`: 0.01–0.3
  - `subsample`: 0.6–1.0
  - `colsample_bytree`: 0.6–1.0
  - `min_child_weight`: 1–50
  - `reg_alpha`: 0–5
  - `reg_lambda`: 0–5
- `n_estimators` handled by early stopping — not a tuning parameter
- n_trials: 50
- Suppress Optuna logging — only print trial progress and best result
- Print best params and tuned MAPE
- Compare tuned vs default MAPE on Fold 1
- Purpose: exploratory — learning which parameters matter and what ranges work

---

### Section 6: Hyperparameter Tuning — Fold 2 (Optuna — Iterative)
- Same Optuna setup as Section 5 but on Fold 2's larger training window
- Warm start: initialize search space around Fold 1 best params
- n_trials: 50 to start — rerun with more trials if not converged
- **This section can and should be rerun freely**
- Each rerun adds trials and refines the search
- Print best params and tuned MAPE after each run
- Print comparison table: default vs tuned MAPE for both folds
- When satisfied with Fold 2 result → write best params as explicit constants
- **FREEZE PARAMS — do not modify after this point**

---

### Section 7: Final Model — Fold 3 (Test)
- Load frozen params from Section 6 constants
- Train on full Feb 2011 → Jan 2015 with frozen params
- Use early stopping on a small held-out slice of training data to set n_estimators
- Predict on Feb 2015 → Jan 2016 — back-transform with expm1
- Evaluate MAPE — this is the number that goes in the master comparison table
- Month-by-month error table
- **Signal ceiling test:** explicitly report errors on Apr 2015, May 2015, Jan 2016
- Compare directly against SARIMA (22.22%) and Prophet (24.25%)
- **Run exactly once. Never rerun after seeing results.**

---

### Section 8: Quantile Models — Fold 3
- Train three XGBoost models on Feb 2011 → Jan 2015 using frozen params:
  - `objective='reg:quantileerror'`, `quantile_alpha=0.10` — conservative
  - `objective='reg:quantileerror'`, `quantile_alpha=0.50` — median
  - `objective='reg:quantileerror'`, `quantile_alpha=0.90` — aggressive
- Plot actual vs 10th/50th/90th percentile bands on representative series
- Confirm 50th percentile MAPE is consistent with point forecast MAPE from Section 7
- These three models feed directly into the Streamlit inventory scenario engine

---

### Section 9: Walk-Forward CV Results Summary
- Master results table:

| Fold | Train Period | Test Period | Default MAPE | Tuned MAPE |
|---|---|---|---|---|
| 1 | Feb 2011 → Jan 2013 | Feb 2013 → Jan 2014 | x% | x% |
| 2 | Feb 2011 → Jan 2014 | Feb 2014 → Jan 2015 | x% | x% |
| 3 | Feb 2011 → Jan 2015 | Feb 2015 → Jan 2016 | — | x% |
| **Avg (Folds 1+2)** | | | x% | x% |

- MAPE improvement from tuning on Folds 1 and 2
- Fold 3 result vs SARIMA (22.22%) and Prophet (24.25%)

---

### Section 10: Signal Ceiling Analysis
- Side-by-side monthly error chart: SARIMA vs Prophet vs XGBoost on representative series
- Highlight Apr 2015, May 2015, Jan 2016 explicitly with annotations
- Quantify the gap closed on each failure month
- Brief interpretation: which features most likely drove the improvement
- This is the headline result of the entire project

---

### Section 11: Global Feature Importance
- XGBoost gain-based importance
- Top 15 features bar chart
- Does the ranking match EDA predictions?
  - Lag features expected to dominate
  - Price and SNAP features expected in top 15 — confirms signal ceiling hypothesis
- Brief interpretation of top features

---

### Section 12: Error Analysis
- MAPE by department — where does the model struggle most?
- MAPE by store — does TX_2 improve over Prophet's 15.31%?
- Error distribution plot — are errors symmetric or skewed?
- Identify the hardest series to forecast and why

---

### Section 13: Save Outputs
- Save `xgb_model.pkl` — the Fold 3 point forecast model
- Save `xgb_quantile_models.pkl` — all three quantile models
- Save `xgb_predictions_val.parquet` — Fold 3 predictions with 10th/50th/90th columns
- Save `xgb_cv_results.csv` — all fold results for master comparison table in notebook 06
- Confirm file sizes and row counts

---

### Section 14: Summary
- What was built and how it was evaluated
- Final CV results table
- Key findings — did we close the signal ceiling?
- Limitations
- What notebook 06 (LightGBM) does next and how results will be compared

---

## Key Rules for This Notebook

| Rule | Reason |
|---|---|
| Optuna runs on Folds 1 and 2 only | Fold 3 test window must never influence any modeling decision |
| Section 6 can be rerun freely | It only looks at Fold 2 val — that is what it is for |
| Section 7 is run exactly once | Seeing the result and then rerunning invalidates the test |
| Fold 2 params written as explicit constants before Section 7 | Makes the freeze visible and auditable |
| Early stopping handles n_estimators | Do not include it in the Optuna search space |
| Back-transform with expm1 before computing MAPE | MAPE on log scale is not interpretable |
| All folds use identical FEATURE_COLS | Guarantees apples-to-apples comparison across folds |
| Quantile models use same frozen params as point model | Consistency across the forecast suite |