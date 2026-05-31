# Forward Plan — Retail Demand Intelligence System
## Strategy B: Full Improvement Pipeline Before Fold 3

**Current state:** 05b validation diagnostics complete. Fold 2 signoff issued.
Fold 3 has never been touched. All improvements below are implemented before
Fold 3 is opened — no leakage risk.

**Discipline:** Fold 3 is run exactly once, at the end of this plan.
Any decision to retrain after seeing Fold 3 results is test set contamination.

---

## Repository Structure (Final)

```
retail-demand-intelligence/
├── data/
│   ├── raw/
│   └── processed/
│       ├── features_train.parquet         ← v1 (frozen, never overwritten)
│       ├── features_val.parquet           ← v1 (frozen, never overwritten)
│       ├── feature_cols.pkl               ← v1
│       ├── features_train_v2.parquet      ← v2 (new features)
│       ├── features_val_v2.parquet        ← v2 (new features)
│       └── feature_cols_v2.pkl            ← v2
├── notebooks/
│   ├── 01_eda.ipynb                       ✅ done
│   ├── 02_baselines_and_stats.ipynb       ✅ done
│   ├── 03_prophet.ipynb                   ✅ done
│   ├── 03b_prophet_stores_conclusion.ipynb ✅ done
│   ├── 04_feature_engineering.ipynb       ✅ done (v1 — frozen)
│   ├── 04b_feature_engineering_v2.ipynb   ← build next
│   ├── 05_xgboost_demand.ipynb            ✅ done (v1 — frozen)
│   ├── 05b_validation_diagnostics.ipynb   ✅ done (v1 — frozen)
│   ├── 05c_xgboost_v2.ipynb               ← new XGBoost on v2 features
│   ├── 05d_validation_diagnostics_v2.ipynb ← v2 fold 2 signoff
│   ├── 06_lightgbm_demand.ipynb           ← LightGBM on v2 features
│   ├── 07_fold3_final_evaluation.ipynb    ← Fold 3, run once, never rerun
│   └── 08_explainability.ipynb            ← SHAP + anomaly detection
├── app.py
├── requirements.txt
└── README.md
```

**Rule:** Notebooks 01–05b are frozen artifacts. They are never modified.
They represent the v1 pipeline and serve as the documented before-state
for the improvement narrative.

---

## Phase 1 — Feature Engineering v2
### Notebook: `04b_feature_engineering_v2.ipynb`

Implements all three feature-level fixes identified in Section 9 of 05b.
Produces `features_train_v2.parquet` and `features_val_v2.parquet` using
identical fold boundaries and gap-aware lag logic as v1.

---

### Fix 1: Pre-Holiday Feature Redesign
**Diagnostic anchor:** Section 9.3. `is_pre_closed_holiday` has mean|SHAP|=0.0000
— functionally dead. Pre-holiday windows show RMSE=0.598, bias=−0.414 across
69K rows. The flat binary fails to encode the continuous demand surge in the
lead-up window.

**Remove:**
- `is_pre_closed_holiday` (binary flag — zero learned contribution)

**Add:**
```python
# Continuous proximity signal — peaks at 14 days out, decays to 0 at 15+
# Gives the model a ramp shape rather than a step function
df['days_to_holiday_proximity'] = (
    (14 - df['days_to_closed_holiday'])
    .clip(lower=0)
    .where(df['days_to_closed_holiday'] > 0, 0)
)

# Category-specific pre-holiday interaction
# FOODS surges 3–5 days before Thanksgiving; HOBBIES surges 7–14 days before Christmas
# A flat binary cannot capture this — a category interaction can
df['preholiday_x_cat'] = df['days_to_holiday_proximity'] * df['cat_id_enc']
```

**Leakage check:** `days_to_closed_holiday` is calendar-derived — no sales
data in the lookback. Safe at all dates.

---

### Fix 2: State-Specific SNAP Cycle Encoding
**Diagnostic anchor:** Section 9.5. ACF lags 14 and 21 show residual bi-weekly
autocorrelation (+0.27, +0.18). SHAP direction for `is_snap` is inverted on
average — the binary flag averages across states with different disbursement
schedules, producing a mixed signal.

**Keep:** `is_snap` (still useful as a base signal)

**Add:**
```python
# SNAP disbursement schedules are publicly documented and fixed across the dataset.
# CA: 1st–10th of month. TX: varies by last digit of case number (approx 1st–15th).
# WI: approx 1st–15th of month.
# Encode day-within-cycle per state — captures within-cycle demand decay.

snap_schedule = {
    'CA': (1, 10),   # disbursement days 1–10
    'TX': (1, 15),
    'WI': (1, 15),
}

def snap_day_of_cycle(row):
    if not row['is_snap']:
        return 0
    state = row['state_id']
    start, end = snap_schedule.get(state, (1, 10))
    day = row['day_of_month']
    if start <= day <= end:
        return day - start + 1   # 1 = first day of cycle (peak purchasing)
    return 0

df['snap_day_of_cycle'] = df.apply(snap_day_of_cycle, axis=1)

# Peak window flag — first 3 days of SNAP disbursement per state
df['is_snap_peak'] = (df['snap_day_of_cycle'] > 0) & (df['snap_day_of_cycle'] <= 3)
df['is_snap_peak'] = df['is_snap_peak'].astype(int)
```

**Expected effect:** Corrects SHAP direction for SNAP, eliminates residual
bi-weekly ACF signal, improves FOODS segment calibration on SNAP days.

---

### Fix 3: sell_price Normalization for Elasticity Recovery
**Diagnostic anchor:** Section 9.5. `sell_price` SHAP pushes demand UP (wrong
direction). Collinearity with product category — premium HOBBIES SKUs have
higher prices and lower demand — causes the tree to learn price as a category
proxy rather than an elasticity signal.

**Keep:** `sell_price` (absolute price still carries signal)

**Add:**
```python
# Price relative to each item's own historical mean — isolates promotional signal
# from category-level price tier. A price below item mean = likely markdown.
item_mean_price = (df.groupby('item_id')['sell_price']
                     .transform('mean')
                     .rename('item_mean_price'))
df['price_vs_item_mean'] = df['sell_price'] / item_mean_price.clip(lower=0.01)

# Rolling 52-week price percentile rank per series
# Captures whether the current price is historically low (promotional floor)
# Lookback is strictly historical — no leakage
df['price_percentile_52w'] = (df.groupby('id')['sell_price']
                                 .transform(lambda x: x.rolling(364, min_periods=28)
                                 .rank(pct=True)))
```

**Leakage check:** `item_mean_price` must be computed from training data only
and joined onto val rows — do not compute from the full dataset. Use the
training period mean, stored as a lookup table, applied to both train and val.
`price_percentile_52w` uses a rolling lookback — safe by construction.

---

### Feature count after v2

| Group | v1 count | v2 count | Change |
|---|---|---|---|
| Temporal | 7 | 7 | — |
| Event/SNAP | 5 | 6 | +`snap_day_of_cycle`, +`is_snap_peak`, −`is_pre_closed_holiday` |
| Pre-holiday | included above | 2 | +`days_to_holiday_proximity`, +`preholiday_x_cat` |
| Price | 5 | 7 | +`price_vs_item_mean`, +`price_percentile_52w` |
| Lag/Rolling | 7 | 7 | — |
| Hierarchical | 4 | 4 | — |
| Categoricals | 6 | 6 | — |
| **Total** | **34** | **39** | **+5** |

### Outputs
- `features_train_v2.parquet`
- `features_val_v2.parquet`
- `feature_cols_v2.pkl`
- `item_mean_price_lookup.pkl` (training-period means, used at inference)

---

## Phase 2 — XGBoost v2 Training and Fold 2 Validation
### Notebook: `05c_xgboost_v2.ipynb`

Re-runs the full XGBoost training pipeline on v2 features. Structure mirrors
`05_xgboost_demand.ipynb` exactly. Same fold boundaries, same evaluation
functions, same three-tier metric system.

**Why re-run Optuna:** The feature space changed. v1 hyperparameters were
optimal for 34 features. With 39 features — including two interaction terms
— the optimal depth, regularization, and column sampling may shift. Running
Optuna again on Fold 2 with v2 features is correct and not leakage (Fold 3
is still untouched).

### Sections

**Section 1 — Setup**
Same constants and eval functions as 05. Add v2 path constants.
Note explicitly: this notebook uses `features_train_v2.parquet`.

**Section 2 — Load v2 Features**
Load, confirm schema matches `feature_cols_v2.pkl`, run null audit.
Print feature count delta vs v1 (34 → 39) as a confirmation step.

**Section 3 — Fold 1 Exploratory Run**
Single run with v1 BEST_PARAMS on v2 features to establish a baseline.
Confirms the new features don't break anything before Optuna runs.

**Section 4 — Optuna Hyperparameter Search (Fold 2)**
Same search space as v1 Optuna. Target: minimize log-RMSE on Fold 2 val.
Run minimum 50 trials. Freeze best params when convergence is confirmed
(no improvement in final 20 trials).

Record: v2 BEST_PARAMS block. Note the delta from v1 params — if depth
or regularization shift substantially, that is evidence the new features
changed the bias-variance tradeoff as expected.

**Section 5 — Fold 2 Retrain with Frozen v2 Params**
Identical to 05 Section 6. Produces predictions for 05d diagnostics.

**Section 6 — Apply Post-Hoc Inference Rules**
These two fixes require no retraining. Apply them here on Fold 2 predictions
before writing outputs — so that 05d evaluates the full v2 pipeline including
inference rules, not just the model in isolation.

```python
# ── Fix A: Hard suppression on closed holiday days ─────────────────────────
# Stores are closed. Sales are structurally impossible. Enforce zero.
closed_mask = va2['is_closed_holiday'].astype(bool)
y_pred_final[closed_mask] = 0.0
y_pred_log[closed_mask]   = 0.0   # also zero out log predictions for metric computation
print(f'Closed holiday suppression applied: {closed_mask.sum():,} rows zeroed.')

# ── Fix B: Zero-rate indexed calibration layer ─────────────────────────────
# Bias correction indexed on per-series zero rate.
# Scale factors derived from Fold 2 residual analysis in 05b Section 8h.
# Applied in log space as an additive correction.
#
# Bucket boundaries and corrections from 05b output:
#   0–10%  zero rate: bias = -0.116 → correction = +0.116
#   10–30%           bias = -0.238 → correction = +0.238
#   30–50%           bias = -0.370 → correction = +0.370
#   50–70%           bias = -0.489 → correction = +0.489
#   70–90%           bias = -0.594 → correction = +0.594
#   90–100%          bias = -0.648 → correction = +0.648
#
# Zero rate is computed from training history only — not from val window.
# Safe to apply at inference. No leakage.

zero_rate_corrections = {
    (0.0,  0.1): 0.116,
    (0.1,  0.3): 0.238,
    (0.3,  0.5): 0.370,
    (0.5,  0.7): 0.489,
    (0.7,  0.9): 0.594,
    (0.9,  1.0): 0.648,
}

# Compute per-series zero rate from training data
train_zero_rate = (tr2.groupby('id')['units_sold']
                      .apply(lambda x: (x == 0).mean())
                      .rename('zero_rate'))

def get_correction(zero_rate):
    for (lo, hi), corr in zero_rate_corrections.items():
        if lo <= zero_rate <= hi:
            return corr
    return 0.0

correction_map = train_zero_rate.apply(get_correction).to_dict()

# Apply correction per row
id_col = va2['id'].values
corrections = np.array([correction_map.get(i, 0.0) for i in id_col])
y_pred_log_calibrated = y_pred_log + corrections
```

**Section 7 — Save Outputs**
- `xgb_v2_model_fold2.pkl`
- `xgb_v2_predictions_fold2.parquet` (includes calibrated predictions)
- `xgb_v2_best_params.pkl`

---

## Phase 3 — v2 Validation Diagnostics and Signoff
### Notebook: `05d_validation_diagnostics_v2.ipynb`

Mirrors `05b` exactly in structure. Evaluates the full v2 pipeline (v2 features
+ v2 params + inference rules) on Fold 2. Issues a formal signoff before
Fold 3 is opened.

**Primary questions this notebook answers:**

| Question | Where answered |
|---|---|
| Does v2 log-RMSE improve on v1 (0.5755)? | Section 4 |
| Does bias improve from −0.372 toward zero? | Section 4 |
| Did holiday suppression fix the closed-holiday segment (was RMSE=0.655)? | Section 6d |
| Did pre-holiday feature redesign improve the pre-holiday segment? | Section 6d |
| Did SNAP features correct the SHAP direction failure? | Section 8e |
| Did price normalization correct sell_price SHAP direction? | Section 8e |
| Did the calibration layer reduce bias monotonically across zero-rate buckets? | Section 8h |
| Did the bi-weekly ACF signal disappear? | Section 5 |
| Is Fold 3 leakage still zero? | Section 2 assertion |

**Abbreviated structure** — same 11 sections as 05b but with a v1 vs v2
comparison table added to Section 4 and Section 11:

```
Section 4 comparison table (add to global performance review):

| Metric          | v1 (05b)  | v2 (05d)  | Delta     |
|-----------------|-----------|-----------|-----------|
| log-RMSE        | 0.5755    | [result]  | [result]  |
| log-MAE         | [v1]      | [result]  | [result]  |
| Bias            | -0.372    | [result]  | [result]  |
| Closed holiday RMSE | 0.655 | [result]  | [result]  |
| Pre-holiday RMSE | 0.598   | [result]  | [result]  |
| Spike RMSE      | 1.371     | [result]  | [result]  |
```

**Signoff gate:** If v2 log-RMSE is within 0.01 of v1 AND bias improves AND
holiday segment improves → APPROVED. If v2 RMSE is worse by more than 0.005
→ investigate before proceeding (one of the new features is adding noise).

---

## Phase 4 — LightGBM
### Notebook: `06_lightgbm_demand.ipynb`

Trains LightGBM on v2 features. Same fold structure, same three-tier evaluation,
same post-hoc inference rules (holiday suppression + calibration layer) applied
identically. This ensures the XGBoost vs LightGBM comparison is apples-to-apples
on the improved pipeline, not on the v1 baseline.

### Sections

**Section 1 — Setup**
LightGBM-specific constants. Import `lightgbm as lgb`. Reuse eval functions
from 05c (same signatures — no changes needed).

**Section 2 — Load v2 Features**
Same parquet files as 05c. Confirm schema.

**Section 3 — Fold 1 Exploratory Run**
Default LightGBM params on v2 features. Establishes a baseline before Optuna.

**Section 4 — Optuna Hyperparameter Search (Fold 2)**
LightGBM search space:
```python
params = {
    'num_leaves':        trial.suggest_int('num_leaves', 64, 512),
    'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
    'feature_fraction':  trial.suggest_float('feature_fraction', 0.5, 1.0),
    'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.5, 1.0),
    'bagging_freq':      trial.suggest_int('bagging_freq', 1, 7),
    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
    'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 1.0),
    'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 1.0),
    'objective':         'regression',
    'metric':            'rmse',
    'verbosity':         -1,
    'n_estimators':      2000,   # controlled by early stopping
}
```
Run minimum 50 trials. Freeze best params on convergence.

**Section 5 — Fold 2 Retrain and Apply Inference Rules**
Same holiday suppression and calibration layer as 05c Section 6.
Zero-rate corrections use the same lookup table (derived from Fold 2
training data — same source for both models, ensuring fair comparison).

**Section 6 — Fold 2 Performance vs XGBoost v2**
Direct comparison table: LightGBM vs XGBoost v2 on every metric from 05d.
This is the primary output of this notebook. If LightGBM wins on Fold 2,
it is the candidate for Fold 3. If XGBoost wins, XGBoost goes to Fold 3.

```
Fold 2 comparison (v2 features + inference rules):

| Metric          | XGBoost v2 | LightGBM  | Winner    |
|-----------------|------------|-----------|-----------|
| log-RMSE        | [05d]      | [result]  |           |
| log-MAE         | [05d]      | [result]  |           |
| Bias            | [05d]      | [result]  |           |
| Closed holiday RMSE | [05d] | [result]  |           |
| Spike RMSE      | [05d]      | [result]  |           |
```

**Section 7 — Winner Selection**
Primary criterion: lower log-RMSE on Fold 2. Tie-break (within 0.005):
lower bias magnitude, then lower closed holiday RMSE.

Document the winner and rationale here. This decision is final — both
models proceed to Fold 3, but the winner's results are the headline numbers.

**Outputs:**
- `lgbm_model_fold2.pkl`
- `lgbm_predictions_fold2.parquet`
- `lgbm_best_params.pkl`
- `model_comparison_fold2.csv` (XGBoost v2 vs LightGBM, all metrics)

---

## Phase 5 — Final Holdout Evaluation
### Notebook: `07_fold3_final_evaluation.ipynb`

**This notebook is run exactly once. Results are never used to make
any modeling decision. The moment Fold 3 results are visible, the
pipeline is locked.**

Both models (XGBoost v2 and LightGBM) are evaluated here under identical
conditions. Post-hoc inference rules are applied identically to both.

### Sections

**Section 1 — Setup and Pre-Flight Checklist**
Before running any cells, confirm in writing:
- [ ] `features_val_v2.parquet` loaded (not features_train)
- [ ] BEST_PARAMS for both models loaded from pkl files — not typed manually
- [ ] `item_mean_price_lookup.pkl` loaded (for price_vs_item_mean at inference)
- [ ] `train_zero_rate` computed from Fold 3 training data (Feb 2011 → Jan 2015)
      — NOT from Fold 2 training data. The calibration layer uses the full
      available training history at Fold 3, which extends one year further.
- [ ] Fold 3 val window: Feb 2015 → Jan 2016
- [ ] Leakage assertion will run before any metric is computed

**Section 2 — Load Fold 3 Data**
Load `features_val_v2.parquet`. This is the first time this file is opened.
Run leakage assertion: confirm max date in train < Feb 2015.
Print shapes, date ranges, series count.

**Section 3 — Retrain Both Models on Full Fold 3 Training Data**
Train XGBoost v2 and LightGBM on Feb 2011 → Jan 2015 (full training history).
Monitor set: last 60 days of training (Dec 2014 → Jan 2015).
Apply frozen params for each model — no Optuna, no adjustments.

**Section 4 — Apply Post-Hoc Inference Rules**
Same rules as 05c and 06. Apply to both models identically:
1. Holiday suppression (closed_holiday rows → 0)
2. Zero-rate calibration (corrections from Fold 3 training data zero rates)

Note: the calibration corrections may differ slightly from Fold 2 values
because the training window is one year longer. Recompute from Fold 3
training data — do not hardcode the Fold 2 numbers.

**Section 5 — Tier 2 Performance: Both Models**
Global log-RMSE, log-MAE, bias for XGBoost v2 and LightGBM.
v1 XGBoost baseline shown for reference (from 05b).

```
Final Fold 3 Results:

| Metric    | XGBoost v1 (baseline) | XGBoost v2 | LightGBM | Winner |
|-----------|-----------------------|------------|----------|--------|
| log-RMSE  | 0.5755 (Fold 2 proxy) | [result]   | [result] |        |
| log-MAE   | —                     | [result]   | [result] |        |
| Bias      | -0.372 (Fold 2 proxy) | [result]   | [result] |        |
```

**Section 6 — Tier 3 Signal Ceiling Comparison**
MAPE on representative series (FOODS_3_163_CA_3), monthly, non-zero months.
Compared directly against SARIMA (22.22%) and Prophet (24.25%).
Focus on Apr 2015, May 2015, Jan 2016 — the three signal ceiling months.

```
Signal Ceiling Results (Tier 3 — representative series, monthly):

| Model     | Overall MAPE | Apr 2015 | May 2015 | Jan 2016 |
|-----------|-------------|----------|----------|----------|
| Naive     | 55.29%      | —        | —        | —        |
| SARIMA    | 22.22%      | [error]  | [error]  | [error]  |
| Prophet   | 24.25%      | [error]  | [error]  | [error]  |
| XGBoost v2| [result]    | [result] | [result] | [result] |
| LightGBM  | [result]    | [result] | [result] | [result] |
```

**Section 7 — Tier 4 Quantile Calibration**
Empirical coverage for q50, q80, q95, q99 on Fold 3 val window.
Both models evaluated. A model is calibrated if empirical coverage
is within ±3 percentage points of the target quantile.

**Section 8 — Segmented Analysis (Abbreviated)**
Key segments only — not the full 05b battery:
- By department (confirm HOBBIES_2 still worst, FOODS_3 still best)
- By demand bucket (confirm bias-volume relationship persists)
- Closed holiday rows (confirm suppression rule zeroed them)
- Signal ceiling months specifically

**Section 9 — Final Model Selection**
Document the winning model with explicit quantitative reasoning.
This is the model used in the Streamlit app and cited in the README.

**Section 10 — Fold 3 Signoff**
One paragraph. State the final numbers. State the winner. State what
the portfolio claims. Lock the notebook.

**Outputs:**
- `xgb_v2_model_fold3.pkl` — final XGBoost model (trained on full Fold 3 train)
- `lgbm_model_fold3.pkl` — final LightGBM model
- `xgb_v2_predictions_fold3.parquet`
- `lgbm_predictions_fold3.parquet`
- `final_model_comparison.csv` — all metrics, all models, all tiers
- `zero_rate_lookup_fold3.pkl` — calibration correction map (from Fold 3 training data)

---

## Phase 6 — Explainability
### Notebook: `08_explainability.ipynb`

Uses the winning model's Fold 3 predictions and the full training data.
SHAP analysis now reflects the v2 feature set — compare directly against
05b Section 8e to confirm the direction failures were corrected.

**Section 1 — SHAP Global Importance**
- Reproduce 05b Section 8e on winning model + v2 features
- Confirm: `is_snap` direction corrected, `sell_price` direction corrected,
  `is_pre_closed_holiday` gone, `days_to_holiday_proximity` has positive SHAP
- Show v1 vs v2 SHAP rank comparison table

**Section 2 — SHAP Direction Validation**
Same direction check table as 05b 8e. All checks should now pass.

**Section 3 — Per-SKU Waterfall Plots**
Three representative series:
- FOODS_3_163_CA_3 (the benchmark series)
- HOBBIES_1_408_CA_2 (worst series from 05b 8h — show what the model sees)
- One SNAP-heavy FOODS series during a SNAP week

**Section 4 — Anomaly Detection**
- Residuals from Fold 3 predictions
- Z-score flags: demand spikes (z > 3), suppressed demand (z < −3)
- Isolation Forest on residual distribution
- Flag the 3,937 anomalous `price_change_pct_raw` rows from feature engineering

---

## Execution Order and Time Estimates

| Step | Notebook | Estimated time | Blocking dependency |
|---|---|---|---|
| 1 | `04b_feature_engineering_v2.ipynb` | 2–3 hours | None — start here |
| 2 | `05c_xgboost_v2.ipynb` (Optuna) | 3–5 hours (GPU) | 04b complete |
| 3 | `05d_validation_diagnostics_v2.ipynb` | 2–3 hours | 05c complete |
| 4 | `05d` signoff decision | — | 05d complete |
| 5 | `06_lightgbm_demand.ipynb` (Optuna) | 3–5 hours (GPU) | 04b complete (parallel with 05c if desired) |
| 6 | Fold 2 model comparison (Section 6 of 06) | 1 hour | 05d + 06 complete |
| 7 | **`07_fold3_final_evaluation.ipynb`** | 2–3 hours | ALL above complete |
| 8 | `08_explainability.ipynb` | 2–3 hours | 07 complete |
| 9 | Streamlit app | 1–2 days | 07 complete |
| 10 | README + portfolio | Half day | All notebooks complete |

**Total modeling work before app:** ~3–4 focused sessions.

---

## Key Decisions Already Made (Do Not Revisit)

| Decision | Rationale |
|---|---|
| v1 notebooks (01–05b) are frozen | They are the documented before-state. Modifying them destroys the improvement narrative. |
| Optuna re-runs on Fold 2 only | Feature space changed — v1 params may not be optimal for v2. Re-running is correct, not overfitting. |
| Both models go to Fold 3 | Winner selected on Fold 2. Fold 3 confirms generalization of both. |
| Calibration corrections recomputed at Fold 3 | The correction map uses training-data zero rates. At Fold 3, training extends one year further — the corrections should use the full available history, not the Fold 2 numbers hardcoded. |
| Fold 3 run once, results locked | Non-negotiable. Any retraining after seeing Fold 3 is leakage regardless of intent. |

---

## What the Portfolio Narrative Looks Like at the End

The repository tells a complete, auditable story in sequence:

1. **EDA** — understand demand structure
2. **Baselines** — establish signal ceiling with SARIMA and Prophet
3. **Feature engineering v1** — first principled attempt
4. **XGBoost v1** — tuned, validated, diagnosed
5. **Diagnostic notebook** — principal-level weakness identification
6. **Feature engineering v2** — structured improvements from diagnostics
7. **XGBoost v2 + LightGBM** — improved pipeline, fair comparison
8. **Fold 3** — one clean final result, never touched before this moment
9. **Explainability** — SHAP confirms the fixes worked
10. **App** — converts forecasts into decisions

Every notebook is a frozen artifact. Every decision is documented and justified.
The before-and-after comparison (v1 → v2) is built into the repository structure.
This is what a production ML team's development history looks like.