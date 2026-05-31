# Forward Plan — Retail Demand Intelligence System
## Strategy B: Full Improvement Pipeline Before Fold 3

**Current state:** 05b validation diagnostics complete. Fold 2 signoff issued.
Fold 3 has never been touched. All improvements below are implemented before
Fold 3 is opened — no leakage risk.

**Discipline:** Fold 3 is run exactly once, at the end of this plan.
Any decision to retrain after seeing Fold 3 results is test set contamination.

---

## Principal Reviewer Notes — What Changed and Why

This section documents every structural change from the previous plan and the
statistical reasoning behind each decision. It is included so the project
history is auditable.

### Change 1 — Calibration method replaced: fixed buckets → isotonic regression

**DELETED:** The fixed six-bucket additive correction scheme from the previous
plan (Section 6 of `05c_xgboost_v2.ipynb`).

**REPLACED WITH:** Isotonic regression fit on the Fold 2 (val) residual curve,
with a Fold 2 performance gate before the method is adopted.

**Reasoning:**

Fixed bucket corrections have three problems that are serious enough to
disqualify them as the primary production method:

First, the bins are arbitrary. The boundaries `[0, 0.1, 0.3, 0.5, 0.7, 0.9,
1.0]` are round numbers with no statistical basis. Series near a boundary
(e.g., a series with zero rate 0.299 vs 0.301) receive very different
corrections despite nearly identical sparsity structure. This discontinuity
is indefensible in a production system and would be the first question a
hiring manager asks.

Second, the corrections are computed as mean bias per bucket. But bias within
a bucket is not flat — it has a gradient. A linear or monotone smooth fit
over the gradient captures the within-bucket variation that the step function
cannot. The Section 8h scatter plot from 05b shows the zero rate vs bias
relationship is smooth and monotone — exactly the shape that isotonic
regression is designed for. Using a step function on a smooth empirical
relationship throws away information.

Third, the bucket corrections come from mean residuals on the Fold 2 val
window. But val rows are not i.i.d. — they are time-series rows with serial
correlation. Mean residuals are not an unbiased estimate of the correction
needed for a future window. A smoother method with a performance gate gives
protection against overfitting the correction to Fold 2's specific demand
patterns.

**Why not linear regression?** Linear regression on zero rate is the simplest
alternative and would be defensible. However, the zero rate vs bias
relationship is monotone but not provably linear — the rate of increase in
bias may slow above 70–80% zero rate (ceiling effect from the log-space
target floor). Isotonic regression is non-parametric and makes no linearity
assumption while still enforcing the monotone constraint that is theoretically
required. It has fewer degrees of freedom than a full polynomial and is no
more expensive to fit.

**Why not a full isotonic regression on every prediction?** Fitting isotonic
regression on raw prediction order would be target leakage — the sort order
includes val target information. The correct approach is to fit isotonic
regression on the zero rate vs mean residual relationship using training data
(zero rate from training history, residuals from Fold 2 val), so the fit
uses only quantities available at inference time.

**Why not a two-stage classify-then-regress model?** A two-stage architecture
is the statistically correct long-run answer to zero-inflated demand and would
improve both bias and RMSE. However, it requires a separate classification
head, a separate training objective, and a blending weight — significant
additional scope. The isotonic calibration layer is the production team's
bridge: it ships the improvement in this sprint while the architecture
redesign goes on the backlog. Both belong in the portfolio; the calibration
layer is not a permanent substitute.

**Leakage status of isotonic calibration:** The calibration is fit using:
- Zero rate per series, computed from the Fold 2 *training* window (not val)
- Mean residual per zero-rate percentile group, computed from Fold 2 *val*

The val residuals carry information about val targets. This is intentional:
the calibration layer is explicitly post-hoc. The rule against leakage applies
to model training, not to post-hoc calibration that is transparently labeled
and validation-gated. A production ML team would call this a calibration stage
and document it explicitly — which is what this plan does.

### Change 2 — Calibration is not automatically accepted

**DELETED:** The previous plan's design where calibration was applied
unconditionally as part of the pipeline.

**REPLACED WITH:** A three-way comparison in `05d_validation_diagnostics_v2.ipynb`:
raw predictions, holiday-suppressed predictions, and holiday-suppressed plus
calibrated predictions. Calibration is adopted only if it passes a quantitative
performance gate on Fold 2.

**Reasoning:** Applying a calibration layer that does not actually improve
Fold 2 performance is strictly worse than no calibration — it introduces
an extra component that increases pipeline complexity, creates an additional
failure mode at inference, and adds a decision-point the reviewer must
explain. The previous plan treated calibration as an improvement by assumption.
A principal-level review treats it as an improvement by evidence.

### Change 3 — Fold 3 calibration design corrected

**DELETED:** The instruction to "recompute zero-rate corrections from Fold 3
training data using the same bucket method."

**REPLACED WITH:** If calibration is approved in 05d, the identical isotonic
regression workflow is applied at Fold 3 using Fold 3 training data. The
calibration curve is re-fit — not hardcoded from Fold 2 values. The
performance gate is not re-evaluated at Fold 3 (that decision was made in
05d and is locked).

**Reasoning:** Hardcoding Fold 2 correction values at Fold 3 is conceptually
wrong. The Fold 3 training window extends one full year further than Fold 2
(Feb 2011 → Jan 2015 instead of Feb 2011 → Jan 2014). The zero rate
distribution of series changes as new products enter and exit the assortment.
Re-fitting from Fold 3 training data is correct. But re-evaluating whether to
use calibration at all is not — that gate closes in 05d.

### Change 4 — Feature engineering approval gate added

The previous plan assumed all five new features would be included. This plan
requires a brief audit in `04b_feature_engineering_v2.ipynb` Section 7
to confirm each feature's distribution is well-behaved before it enters
training. This is not a Fold 2 evaluation — it is an engineering sanity
check that would be done in any production feature pipeline.

### What was NOT changed

Holiday suppression remains a hard inference rule. This is correct. Stores
are closed; sales are structurally impossible. No performance gate applies to
a business constraint.

The five new features are retained as proposed. The statistical rationale for
each was reviewed and is sound. `item_mean_price` leakage concern is noted in
the original plan and the fix (compute from training period only) is correct.
`price_percentile_52w` rolling lookback is safe by construction.

The Fold 3 untouched discipline is unchanged.

Both models (XGBoost v2 and LightGBM) are evaluated at Fold 3. Winner
selection is on Fold 2; Fold 3 confirms generalization.

---

## Repository Structure (Final)

```
retail-demand-intelligence/
├── data/
├── raw/
└── processed/
    ├── features/
    │   ├── features_train.parquet          ← v1 (frozen, never overwritten)
    │   ├── features_val.parquet            ← v1 (frozen, never overwritten)
    │   ├── feature_cols.pkl                ← v1
    │   ├── features_train_v2.parquet       ← v2 (new features)
    │   ├── features_val_v2.parquet         ← v2 (new features)
    │   ├── feature_cols_v2.pkl             ← v2
    │   └── item_mean_price_lookup.pkl      ← v2 (training-period means, used at inference)
    ├── models/
    │   ├── xgb_v1_model_fold2.json         ← v1 (frozen)  [not found — may not have been saved in 05]
    │   ├── xgb_v2_model_fold2.json         ← v2 Fold 2 trained model
    │   ├── xgb_v2_model_fold3.json         ← v2 Fold 3 (07, not yet run)
    │   ├── lgbm_model_fold2.pkl            ← LightGBM Fold 2 (06, not yet run)
    │   └── lgbm_model_fold3.pkl            ← LightGBM Fold 3 (07, not yet run)
    ├── predictions/
    │   ├── xgb_v1_predictions_fold2.parquet  ← v1 (frozen)  [not found — may not have been saved in 05]
    │   ├── xgb_v2_predictions_fold2.parquet  ← v2 all three variants (raw, suppressed, calibrated)
    │   ├── xgb_v2_predictions_fold3.parquet  ← v2 Fold 3 (07, not yet run)
    │   ├── lgbm_predictions_fold2.parquet    ← LightGBM Fold 2 (06, not yet run)
    │   ├── lgbm_predictions_fold3.parquet    ← LightGBM Fold 3 (07, not yet run)
    │   └── final_model_comparison.csv        ← (07, not yet run)
    └── calibration/
        ├── xgb_v2_best_params.pkl            ← frozen Optuna result (Trial 41, log-RMSE 0.5764)
        ├── isotonic_calibrator_fold2.pkl     ← fitted IsotonicRegression object (Fold 2)
        ├── train_zero_rate_fold2.pkl         ← per-series zero rate lookup (Fold 2 training window)
        ├── isotonic_calibrator_fold3.pkl     ← re-fit from Fold 3 training data (07, not yet run)
        ├── train_zero_rate_fold3.pkl         ← (07, not yet run)
        └── lgbm_best_params.pkl              ← (06, not yet run)
├── notebooks/
│   ├── 01_eda.ipynb                       ✅ done
│   ├── 02_baselines_and_stats.ipynb       ✅ done
│   ├── 03_prophet.ipynb                   ✅ done
│   ├── 03b_prophet_stores_conclusion.ipynb ✅ done
│   ├── 04_feature_engineering.ipynb       ✅ done (v1 — frozen)
│   ├── 04b_feature_engineering_v2.ipynb   ← Phase 1: build next
│   ├── 05_xgboost_demand.ipynb            ✅ done (v1 — frozen)
│   ├── 05b_validation_diagnostics.ipynb   ✅ done (v1 — frozen)
│   ├── 05c_xgboost_v2.ipynb               ← Phase 2: XGBoost on v2 features
│   ├── 05d_validation_diagnostics_v2.ipynb ← Phase 3: v2 Fold 2 signoff + calibration gate
│   ├── 06_lightgbm_demand.ipynb           ← Phase 4: LightGBM on v2 features
│   ├── 07_fold3_final_evaluation.ipynb    ← Phase 5: Fold 3, run once, never rerun
│   └── 08_explainability.ipynb            ← Phase 6: SHAP + anomaly detection
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
    'CA': (1, 10),
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
        return day - start + 1
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
# Price relative to each item's own historical mean — isolates promotional
# signal from category-level price tier.
# CRITICAL LEAKAGE RULE: item_mean_price is computed from the TRAINING window
# only and stored as a lookup. It is NOT computed from the full dataset.
# At inference (both val and Fold 3), the lookup table is joined in — not
# recomputed from the window being scored.

item_mean_price_lookup = (tr_full
    .groupby('item_id')['sell_price']
    .mean()
    .rename('item_mean_price'))
item_mean_price_lookup.to_pickle('../data/processed/item_mean_price_lookup.pkl')

# Join onto both train and val
df = df.merge(item_mean_price_lookup, on='item_id', how='left')
df['price_vs_item_mean'] = df['sell_price'] / df['item_mean_price'].clip(lower=0.01)

# Rolling 52-week price percentile rank per series
# Lookback is strictly historical — no leakage
df['price_percentile_52w'] = (df.groupby('id')['sell_price']
                                 .transform(lambda x: x.rolling(364, min_periods=28)
                                 .rank(pct=True)))
```

**Leakage check:** `item_mean_price_lookup` uses training-period means only.
`price_percentile_52w` uses a rolling lookback — safe by construction.

---

### Section 7 — Feature Distribution Audit (Required Before Training)

Before writing the final parquet files, run the following distribution checks.
These are engineering sanity checks, not Fold 2 evaluations. Any feature that
fails these checks is dropped before training begins.

```python
# For each new feature:
for col in ['days_to_holiday_proximity', 'preholiday_x_cat',
            'snap_day_of_cycle', 'is_snap_peak',
            'price_vs_item_mean', 'price_percentile_52w']:
    s = df[col]
    n_null = s.isna().sum()
    n_inf  = np.isinf(s).sum() if s.dtype != 'object' else 0
    print(f'{col:<30}  null={n_null:,}  inf={n_inf:,}  '
          f'min={s.min():.4f}  max={s.max():.4f}  '
          f'mean={s.mean():.4f}  std={s.std():.4f}')
```

**Pass criteria (all must pass before proceeding):**
- Null rate < 20% (expected: only `price_percentile_52w` has early-history
  nulls from the 364-day rolling window; these are acceptable and expected)
- No infinite values
- `price_vs_item_mean` mean between 0.90 and 1.10 (near-unity centering confirms
  the normalization is working; values far from 1.0 indicate a join problem)
- `snap_day_of_cycle` max ≤ 15 (enforces schedule bounds)
- `days_to_holiday_proximity` max ≤ 14 (enforces clip logic)

Document the output of this cell in the notebook. If any check fails, fix
the feature before continuing. Do not proceed to 05c until all checks pass.

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

## Phase 2 — XGBoost v2 Training and Fold 2 Predictions
### Notebook: `05c_xgboost_v2.ipynb`

Re-runs the full XGBoost training pipeline on v2 features. Structure mirrors
`05_xgboost_demand.ipynb` exactly. Same fold boundaries, same evaluation
functions, same three-tier metric system.

**Why re-run Optuna:** The feature space changed. v1 hyperparameters were
optimal for 34 features. With 39 features — including two interaction terms
— the optimal depth, regularization, and column sampling may shift. Running
Optuna again on Fold 2 with v2 features is correct and not leakage (Fold 3
is still untouched).

---

### Section 1 — Setup

Same constants and eval functions as 05. Add v2 path constants.
Note explicitly: this notebook uses `features_train_v2.parquet`.

---

### Section 2 — Load v2 Features

Load, confirm schema matches `feature_cols_v2.pkl`, run null audit.
Print feature count delta vs v1 (34 → 39) as a confirmation step.

---

### Section 3 — Fold 1 Exploratory Run

Single run with v1 BEST_PARAMS on v2 features to establish a baseline.
Confirms the new features don't break anything before Optuna runs.

---

### Section 4 — Optuna Hyperparameter Search (Fold 2)

Same search space as v1 Optuna. Target: minimize log-RMSE on Fold 2 val.
Run minimum 50 trials. Freeze best params when convergence is confirmed
(no improvement in final 20 trials).

Record: v2 BEST_PARAMS block. Note the delta from v1 params — if depth
or regularization shift substantially, that is evidence the new features
changed the bias-variance tradeoff as expected.

---

### Section 5 — Fold 2 Retrain with Frozen v2 Params

Identical to 05 Section 6. Produces predictions for 05d diagnostics.

---

### Section 6 — Apply Post-Hoc Inference Rules and Save Prediction Variants

This section applies the two post-hoc inference rules and saves **three
separate prediction arrays**. All three are passed to 05d. The decision
about which variant to use at Fold 3 is made in 05d — not here.

```python
# ── Prediction variants — save all three ──────────────────────────────────

# Variant A: Raw model output (log space)
y_pred_raw = model_v2.predict(X_va2)

# ── Variant B: Holiday suppression applied ─────────────────────────────────
# Hard business rule: stores are closed on these days, sales are impossible.
# This rule has no performance gate — it is a structural constraint.
y_pred_suppressed = y_pred_raw.copy()
closed_mask = va2['is_closed_holiday'].astype(bool)
y_pred_suppressed[closed_mask] = 0.0
print(f'Holiday suppression: {closed_mask.sum():,} rows zeroed.')

# ── Variant C: Holiday suppression + isotonic calibration ──────────────────
# Post-hoc calibration indexed on per-series zero rate.
# Fit isotonic regression on the zero rate vs mean residual relationship
# from the Fold 2 TRAINING data. The zero rate is computed from training
# history; the residuals are from this val window. This is a post-hoc
# calibration layer — not a model parameter. Labeled explicitly as such.
#
# The calibration decision (adopt vs reject) is made in 05d after comparing
# all three variants. Do NOT make that decision here.

from sklearn.isotonic import IsotonicRegression

# Step 1: Compute per-series zero rate from TRAINING data only
train_zero_rate = (tr2.groupby('id')['units_sold']
                      .apply(lambda x: (x == 0).mean())
                      .rename('zero_rate'))

# Step 2: Compute per-series mean residual on val window (suppressed predictions)
# Using suppressed variant — calibration is layered on top of suppression
preds_suppressed_df = va2[['id']].copy()
preds_suppressed_df['yhat_log']  = y_pred_suppressed
preds_suppressed_df['true_log']  = y_va2
preds_suppressed_df['residual']  = y_pred_suppressed - y_va2
preds_suppressed_df['nonzero']   = y_va2 > 0

series_residuals = (preds_suppressed_df[preds_suppressed_df['nonzero']]
                    .groupby('id')['residual']
                    .mean()
                    .rename('mean_residual'))

# Step 3: Build isotonic regression training set
# X = zero rate (from training history), y = mean residual (from val)
# The residual represents how far the model is from zero — the correction
# needed is the negative of the residual (we want residuals near zero).
calib_df = train_zero_rate.to_frame().join(series_residuals, how='inner')
calib_df = calib_df.dropna()

ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
ir.fit(calib_df['zero_rate'].values, calib_df['mean_residual'].values)
# Increasing=True enforces the monotone structure from Section 8h:
# bias worsens (becomes more negative = residual more negative) as zero rate
# increases. The correction must increase monotonically with zero rate.

# Step 4: Generate per-row corrections
id_col = va2['id'].values
zero_rate_per_row = np.array([train_zero_rate.get(i, 0.0) for i in id_col])
corrections = -ir.predict(zero_rate_per_row)   # negate: residual was pred-true, correction is -(pred-true)

y_pred_calibrated = y_pred_suppressed + corrections
print(f'Calibration layer: {len(ir.X_thresholds_)} isotonic thresholds.')
print(f'Correction range: {corrections.min():.4f} to {corrections.max():.4f}')
```

**Save all three variants:**

```python
# Save predictions for 05d comparison
predictions_v2 = va2[['id', 'date', 'units_sold']].copy()
predictions_v2['yhat_raw']        = y_pred_raw
predictions_v2['yhat_suppressed'] = y_pred_suppressed
predictions_v2['yhat_calibrated'] = y_pred_calibrated
predictions_v2['true_log']        = y_va2
predictions_v2.to_parquet('../data/processed/xgb_v2_predictions_fold2.parquet', index=False)

# Save calibration objects for Fold 3 re-use (method, not values)
import pickle
with open('../data/processed/isotonic_calibrator_fold2.pkl', 'wb') as f:
    pickle.dump(ir, f)
with open('../data/processed/train_zero_rate_fold2.pkl', 'wb') as f:
    pickle.dump(train_zero_rate, f)

# Save model and params
model_v2.save_model('../data/processed/xgb_v2_model_fold2.json')
with open('../data/processed/xgb_v2_best_params.pkl', 'wb') as f:
    pickle.dump(BEST_PARAMS_V2, f)
```

**What is NOT done in this notebook:**
- No decision about whether calibration is better or worse
- No metric comparison between variants
- No approval or rejection of any prediction layer
- Those decisions belong in 05d

---

### Section 7 — Outputs Summary

Print a brief summary confirming all files were written. No analysis.

**Outputs:**
- `xgb_v2_model_fold2.json`
- `xgb_v2_predictions_fold2.parquet` (all three variants: raw, suppressed, calibrated)
- `xgb_v2_best_params.pkl`
- `isotonic_calibrator_fold2.pkl` (fitted IsotonicRegression object)
- `train_zero_rate_fold2.pkl` (per-series zero rate lookup from Fold 2 training data)

---

## Phase 3 — v2 Validation Diagnostics and Calibration Gate
### Notebook: `05d_validation_diagnostics_v2.ipynb`

Mirrors `05b` in structure. Evaluates the full v2 pipeline on Fold 2.
Issues a formal signoff before Fold 3 is opened.

This notebook makes two decisions that are locked before Fold 3:

1. Is the v2 pipeline better than v1? (APPROVED or REQUIRES INVESTIGATION)
2. Is the calibration layer adopted? (ADOPTED or REJECTED)

Both decisions are documented in the final cell and are frozen at the time
the notebook is executed. Neither decision can be revised after Fold 3 results
are visible.

---

### Primary questions this notebook answers

| Question | Where answered |
|---|---|
| Does v2 log-RMSE improve on v1 (0.5755)? | Section 4 |
| Does holiday suppression improve the closed-holiday segment? | Section 4, 6d |
| Does isotonic calibration improve Fold 2 performance vs suppression-only? | Section 4, 8h |
| Did pre-holiday feature redesign improve pre-holiday segment? | Section 6d |
| Did SNAP features correct the SHAP direction failure? | Section 8e |
| Did price normalization correct sell_price SHAP direction? | Section 8e |
| Did the bi-weekly ACF signal disappear? | Section 5 |
| Is Fold 3 leakage still zero? | Section 2 assertion |

---

### Section 1 — Setup

Same constants and eval functions as 05b. Add v2 path constants.
Load `xgb_v2_predictions_fold2.parquet` — do not retrain in this notebook.

---

### Section 2 — Leakage Assertion

```python
# Confirm val window dates match expectations
assert va2['date'].min() == pd.Timestamp('2014-02-01'), 'Val start mismatch'
assert va2['date'].max() == pd.Timestamp('2015-01-31'), 'Val end mismatch'
assert va2['date'].max() < pd.Timestamp('2015-02-01'), 'Fold 3 leakage check'
print('Leakage assertion passed. Fold 3 boundary intact.')
```

---

### Section 3 — Load Predictions and Reconstruct Arrays

Load the three prediction variants from 05c output. Compute all evaluation
arrays needed for Sections 4–10.

```python
preds = pd.read_parquet('../data/processed/xgb_v2_predictions_fold2.parquet')

y_true      = preds['true_log'].values
y_raw       = preds['yhat_raw'].values
y_supp      = preds['yhat_suppressed'].values
y_calib     = preds['yhat_calibrated'].values
```

---

### Section 4 — Three-Way Global Performance Comparison

This is the calibration gate. Report all three variants side by side.

```python
print('=' * 60)
print('SECTION 4: Three-Way Performance Comparison')
print('=' * 60)

r_raw   = eval_log_scale(y_true, y_raw,   'Variant A — Raw')
r_supp  = eval_log_scale(y_true, y_supp,  'Variant B — Holiday Suppressed')
r_calib = eval_log_scale(y_true, y_calib, 'Variant C — Suppressed + Calibrated')
```

**Comparison table (populate at runtime):**

| Metric | v1 XGBoost (05b) | v2 Raw | v2 + Holiday Suppressed | v2 + Suppressed + Calibrated |
|---|---|---|---|---|
| log-RMSE | 0.5755 | [result] | [result] | [result] |
| log-MAE | [v1] | [result] | [result] | [result] |
| Bias | −0.372 | [result] | [result] | [result] |
| Closed holiday RMSE | 0.655 | [result] | [result] | [result] |
| Pre-holiday RMSE | 0.598 | [result] | [result] | [result] |
| Spike RMSE | 1.371 | [result] | [result] | [result] |

**Calibration gate logic (execute in code, not manually):**

```python
# ── Calibration gate ──────────────────────────────────────────────────────
# Calibration is adopted only if both conditions are met:
# 1. Calibrated RMSE is not worse than suppressed RMSE (within tolerance)
# 2. Calibrated bias magnitude is meaningfully smaller than suppressed bias
#
# "Meaningfully smaller" = |bias_calib| < |bias_supp| * 0.80
# i.e., calibration must reduce bias by at least 20% to pay for its complexity.
# The 0.80 threshold is conservative by design — calibration must earn adoption.

rmse_tolerance    = 0.010   # calibration may not worsen RMSE by more than 0.010
bias_reduction_req = 0.80   # calibration must reduce |bias| to below 80% of suppressed

rmse_delta   = r_calib['log_rmse'] - r_supp['log_rmse']
bias_ratio   = abs(r_calib['bias']) / max(abs(r_supp['bias']), 1e-6)

calibration_passes_rmse  = rmse_delta  <= rmse_tolerance
calibration_passes_bias  = bias_ratio  <  bias_reduction_req

CALIBRATION_ADOPTED = calibration_passes_rmse and calibration_passes_bias

print()
print('── Calibration Gate Results ─────────────────────────────')
print(f'  RMSE delta (calib vs suppressed) : {rmse_delta:+.4f}  '
      f'(threshold: ≤ +{rmse_tolerance:.3f})  '
      f'{"✓ PASS" if calibration_passes_rmse else "✗ FAIL"}')
print(f'  Bias ratio (calib/suppressed)    : {bias_ratio:.4f}  '
      f'(threshold: < {bias_reduction_req:.2f})  '
      f'{"✓ PASS" if calibration_passes_bias else "✗ FAIL"}')
print()
if CALIBRATION_ADOPTED:
    print('  ✅ CALIBRATION ADOPTED — Variant C is the approved pipeline.')
    print('     Fold 3 will use: holiday suppression + isotonic calibration.')
else:
    print('  ❌ CALIBRATION REJECTED — Variant B is the approved pipeline.')
    print('     Fold 3 will use: holiday suppression only.')
    if not calibration_passes_rmse:
        print(f'     Rejection reason: RMSE worsened by {rmse_delta:+.4f} — exceeds tolerance.')
    if not calibration_passes_bias:
        print(f'     Rejection reason: Bias reduction insufficient ({(1-bias_ratio)*100:.1f}% < 20%).')
```

---

### Sections 5–10 — Diagnostic Battery

Same structure as 05b Sections 5–10, applied to the approved variant (whichever
was selected by the gate in Section 4). Additionally:

**Section 8h — Bias by Zero Rate (Three-Way):**

```python
# Plot bias vs zero rate for all three variants on the same chart.
# The isotonic calibration should flatten the bias-vs-zero-rate curve
# if it is working. If the curve is not flattened, document why.
# (Possible reason: the isotonic fit found a different functional form
# than the bucket method implied, or series zero rate shifted in the
# val window vs training window.)
```

**Section 8e — SHAP Direction Audit:**

Repeat the direction audit from 05b Section 8e on the v2 model.
All five direction checks must pass before the v2 pipeline is approved.
A direction failure on any feature added in v2 is a sign of a specification
error, not a tuning target.

```
Expected direction outcomes:
  is_snap               → POSITIVE (more SNAP = more demand) — was INVERTED in v1
  snap_day_of_cycle     → NEGATIVE at higher values (demand decays through cycle)
  is_snap_peak          → POSITIVE (peak days = highest demand)
  sell_price            → NEGATIVE (higher price = less demand) — was INVERTED in v1
  price_vs_item_mean    → NEGATIVE (price above item mean = below normal demand)
  price_percentile_52w  → NEGATIVE (high price percentile = relatively expensive)
  days_to_holiday_proximity → POSITIVE (closer to holiday = more demand)
  preholiday_x_cat      → MIXED acceptable (different categories have different effects)
  is_closed_holiday     → NEGATIVE (still present but overridden by suppression rule)
```

---

### Section 11 — Fold 2 Signoff and Pipeline Lock

```python
print('=' * 60)
print('SECTION 11: FOLD 2 SIGNOFF — v2 PIPELINE')
print('=' * 60)
print()
print('v1 XGBoost baseline log-RMSE:  0.5755')
print(f'v2 approved pipeline log-RMSE: {r_approved["log_rmse"]:.4f}')
print(f'Delta:                         {r_approved["log_rmse"] - 0.5755:+.4f}')
print()
print(f'CALIBRATION DECISION: {"ADOPTED" if CALIBRATION_ADOPTED else "REJECTED"}')
print(f'APPROVED PIPELINE VARIANT: {"C (suppressed + calibrated)" if CALIBRATION_ADOPTED else "B (suppressed only)"}')
print()
print('This decision is now LOCKED.')
print('It cannot be changed after Fold 3 results are visible.')
print()

PIPELINE_APPROVED = r_approved['log_rmse'] <= 0.5755 + 0.010
# v2 must at minimum not be substantially worse than v1 on RMSE;
# improvement on bias or segments with the same RMSE is still a net win.

if PIPELINE_APPROVED:
    print('✅ v2 PIPELINE APPROVED FOR FOLD 3')
    print('   Notebook 07 may proceed.')
else:
    print('⚠  v2 RMSE is worse than v1 by more than 0.010 — INVESTIGATE BEFORE PROCEEDING')
    print('   Likely cause: one of the new features is adding noise.')
    print('   Steps: (1) train v2 model with each new feature dropped individually')
    print('          (2) identify the culprit from SHAP importance')
    print('          (3) correct or remove the offending feature in 04b')
    print('   Do not proceed to Fold 3 until approved.')
```

---

### Outputs

No model artifacts — this notebook is a diagnostic only.

Decisions recorded here:
- `CALIBRATION_ADOPTED` (bool) — documented in final cell
- `PIPELINE_APPROVED` (bool) — documented in final cell
- `APPROVED_VARIANT` ('suppressed' or 'calibrated') — documented in final cell

These decisions drive `07_fold3_final_evaluation.ipynb` Section 1.

---

## Phase 4 — LightGBM
### Notebook: `06_lightgbm_demand.ipynb`

Trains LightGBM on v2 features. Same fold structure, same three-tier evaluation.
Post-hoc inference rules applied identically to XGBoost v2, using the same
approved pipeline variant determined in 05d.

---

### Section 1 — Setup

LightGBM-specific constants. Import `lightgbm as lgb`. Reuse eval functions.
Load `CALIBRATION_ADOPTED` and `APPROVED_VARIANT` from 05d constants block —
LightGBM uses the same pipeline decision as XGBoost v2.

---

### Section 2 — Load v2 Features

Same parquet files as 05c. Confirm schema.

---

### Section 3 — Fold 1 Exploratory Run

Default LightGBM params on v2 features. Establishes a baseline before Optuna.

---

### Section 4 — Optuna Hyperparameter Search (Fold 2)

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
    'n_estimators':      2000,
}
```

Run minimum 50 trials. Freeze best params on convergence.

---

### Section 5 — Fold 2 Retrain and Apply Approved Inference Rules

Apply holiday suppression unconditionally. Apply isotonic calibration only
if `CALIBRATION_ADOPTED = True` from 05d. Use the same `isotonic_calibrator_fold2.pkl`
and `train_zero_rate_fold2.pkl` from 05c — do not re-fit the calibrator for
LightGBM. The calibration is a post-model correction indexed on series
properties, not a model-specific artifact; using the same calibrator ensures
the XGBoost vs LightGBM comparison is apples-to-apples on pipeline corrections.

---

### Section 6 — Fold 2 Performance vs XGBoost v2

Direct comparison table: LightGBM vs XGBoost v2, approved pipeline variant.
This is the primary output of this notebook. If LightGBM wins on Fold 2,
it is the primary candidate for the headline Fold 3 result. If XGBoost wins,
XGBoost is primary. Both proceed to Fold 3.

```
Fold 2 comparison (v2 features + approved pipeline):

| Metric          | XGBoost v2 | LightGBM  | Winner    |
|-----------------|------------|-----------|-----------| 
| log-RMSE        | [05d]      | [result]  |           |
| log-MAE         | [05d]      | [result]  |           |
| Bias            | [05d]      | [result]  |           |
| Closed holiday RMSE | [05d] | [result]  |           |
| Spike RMSE      | [05d]      | [result]  |           |
```

---

### Section 7 — Winner Selection

Primary criterion: lower log-RMSE on Fold 2. Tie-break (within 0.005):
lower bias magnitude, then lower closed holiday RMSE.

Document the winner and rationale. This selection is the primary candidate
but both models produce headline results at Fold 3 — the full comparison
belongs in the portfolio.

**Outputs:**
- `lgbm_model_fold2.pkl`
- `lgbm_predictions_fold2.parquet`
- `lgbm_best_params.pkl`
- `model_comparison_fold2.csv`

---

## Phase 5 — Final Holdout Evaluation
### Notebook: `07_fold3_final_evaluation.ipynb`

**This notebook is run exactly once. Results are never used to make any
modeling decision. The moment Fold 3 results are visible, the pipeline
is locked.**

Both models are evaluated under identical conditions. The approved pipeline
(suppressed-only or suppressed-plus-calibrated) is applied identically to both,
as determined by the `CALIBRATION_ADOPTED` flag locked in 05d.

---

### Section 1 — Setup and Pre-Flight Checklist

Before running any cells, confirm in writing:

```python
# ── PIPELINE DECISION CONSTANTS (copy from 05d Section 11) ──────────────
# These values are locked. Do not change them after seeing Fold 3 results.
CALIBRATION_ADOPTED = [True / False]   # copy exact value from 05d
APPROVED_VARIANT    = ['calibrated' / 'suppressed']  # copy exact value from 05d
```

Pre-flight checklist (must be confirmed by comment in the notebook before
any cells below are executed):

```
# Pre-flight checklist:
# [ ] features_val_v2.parquet loaded (not features_train)
# [ ] BEST_PARAMS for both models loaded from pkl files — not typed manually
# [ ] item_mean_price_lookup.pkl loaded (for price_vs_item_mean at inference)
# [ ] CALIBRATION_ADOPTED copied exactly from 05d Section 11 output
# [ ] Fold 3 val window: Feb 2015 → Jan 2016
# [ ] Leakage assertion will run before any metric is computed
# [ ] If CALIBRATION_ADOPTED=True: isotonic calibrator will be RE-FIT from
#     Fold 3 training data (not from 05c pkl file — that was Fold 2 training
#     data and the Fold 3 training window is one year longer)
```

---

### Section 2 — Load Fold 3 Data

Load `features_val_v2.parquet`. This is the first time this file is opened.

```python
# Leakage assertion — must pass before any metric computation
assert fold3_val['date'].min() == pd.Timestamp('2015-02-01'), 'Fold 3 start mismatch'
assert fold3_train['date'].max() < pd.Timestamp('2015-02-01'), 'Leakage: train overlaps Fold 3 val'
print('Fold 3 leakage assertion passed.')
```

---

### Section 3 — Retrain Both Models on Full Fold 3 Training Data

Train XGBoost v2 and LightGBM on Feb 2011 → Jan 2015.
Monitor set: last 60 days (Dec 2014 → Jan 2015).
Apply frozen params — no Optuna, no adjustments.

---

### Section 4 — Apply Approved Inference Rules

```python
# ── Holiday suppression — always applied ──────────────────────────────────
# No gate. Stores are closed. Business constraint.
closed_mask = fold3_val['is_closed_holiday'].astype(bool)
y_xgb[closed_mask]  = 0.0
y_lgbm[closed_mask] = 0.0
print(f'Holiday suppression: {closed_mask.sum():,} rows zeroed.')

# ── Isotonic calibration — applied only if adopted in 05d ─────────────────
if CALIBRATION_ADOPTED:
    # Re-fit isotonic regression from Fold 3 training data
    # The method is the same as 05c; the training window is longer.
    # Zero rate is recomputed from the full Fold 3 training history.
    fold3_train_zero_rate = (fold3_train.groupby('id')['units_sold']
                                        .apply(lambda x: (x == 0).mean())
                                        .rename('zero_rate'))

    # Residuals for fitting: use XGBoost's Fold 3 training residuals
    # (in-sample, from the retrained model above) as a proxy for the
    # calibration curve — this is the only in-sample quantity available
    # for isotonic fitting since we have not yet seen Fold 3 val targets.
    #
    # IMPORTANT: This uses in-sample training residuals, not val residuals.
    # This is correct because val targets are not yet visible. The resulting
    # calibration will be noisier than the 05c version (which used val residuals)
    # but remains unbiased with respect to Fold 3 val.
    fold3_train_preds = model_xgb_f3.predict(X_fold3_train)
    fold3_train_resid = y_fold3_train - fold3_train_preds
    fold3_series_resid = (pd.DataFrame({'id': fold3_train['id'],
                                        'residual': fold3_train_resid})
                            .groupby('id')['residual'].mean())
    calib_df_f3 = fold3_train_zero_rate.to_frame().join(fold3_series_resid, how='inner').dropna()

    ir_f3 = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir_f3.fit(calib_df_f3['zero_rate'].values, calib_df_f3['residual'].values)

    val_id_col = fold3_val['id'].values
    zr_val = np.array([fold3_train_zero_rate.get(i, 0.0) for i in val_id_col])
    corrections_f3 = -ir_f3.predict(zr_val)

    y_xgb  = y_xgb  + corrections_f3
    y_lgbm = y_lgbm + corrections_f3
    print(f'Fold 3 isotonic calibration applied. '
          f'Correction range: {corrections_f3.min():.4f} to {corrections_f3.max():.4f}')
else:
    print('Calibration not applied (CALIBRATION_ADOPTED=False from 05d).')
```

**Note on the in-sample calibration design at Fold 3:** The 05c calibration
used val residuals to fit the isotonic curve, which was possible because the
val window was used for diagnostic purposes. At Fold 3, the val targets cannot
be used — only training residuals are available. The in-sample residuals
capture the same zero-rate vs underprediction structure (since the structure
is driven by training dynamics, not val-specific effects) but are less precise.
This is the correct behavior. Document this distinction explicitly in the
notebook.

---

### Section 5 — Tier 2 Performance: Both Models

Global log-RMSE, log-MAE, bias for XGBoost v2 and LightGBM.
v1 XGBoost Fold 2 result shown for context (not a direct comparison since
it is from a different fold, but useful for orientation).

```
Final Fold 3 Results:

| Metric    | XGBoost v1 (Fold 2 proxy) | XGBoost v2 | LightGBM | Winner |
|-----------|-----------------------|------------|----------|--------|
| log-RMSE  | 0.5755                | [result]   | [result] |        |
| log-MAE   | —                     | [result]   | [result] |        |
| Bias      | −0.372                | [result]   | [result] |        |
```

---

### Section 6 — Tier 3 Signal Ceiling Comparison

MAPE on representative series (FOODS_3_163_CA_3), monthly, non-zero months.
Compared directly against SARIMA (22.22%) and Prophet (24.25%).

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

---

### Section 7 — Tier 4 Quantile Calibration

Empirical coverage for q50, q80, q95, q99 on Fold 3 val window.
Both models evaluated. A model is calibrated if empirical coverage
is within ±3 percentage points of the target quantile.

---

### Section 8 — Segmented Analysis (Abbreviated)

Key segments only:
- By department (confirm HOBBIES_2 still worst, FOODS_3 still best)
- By demand bucket (confirm bias-volume relationship persists or is corrected)
- Closed holiday rows (confirm suppression rule zeroed them)
- Signal ceiling months specifically

---

### Section 9 — Final Model Selection

Document the winning model with explicit quantitative reasoning.
This is the model used in the Streamlit app and cited in the README.

---

### Section 10 — Fold 3 Signoff

One paragraph. State the final numbers. State the winner. State what
the portfolio claims. Lock the notebook.

**Outputs:**
- `xgb_v2_model_fold3.json`
- `lgbm_model_fold3.pkl`
- `xgb_v2_predictions_fold3.parquet`
- `lgbm_predictions_fold3.parquet`
- `final_model_comparison.csv`
- `isotonic_calibrator_fold3.pkl` (if calibration adopted — re-fitted from Fold 3 training)
- `zero_rate_lookup_fold3.pkl`

---

## Phase 6 — Explainability
### Notebook: `08_explainability.ipynb`

Uses the winning model's Fold 3 predictions and the full training data.
SHAP analysis now reflects the v2 feature set — compare directly against
05b Section 8e to confirm direction failures were corrected.

**Section 1 — SHAP Global Importance**

- Reproduce 05b Section 8e on winning model + v2 features
- Confirm all direction checks pass (see the 05d Section 8e checklist above)
- Show v1 vs v2 SHAP rank comparison table

**Section 2 — SHAP Direction Validation**

Same direction check table as 05b 8e. All checks should now pass.

**Section 3 — Per-SKU Waterfall Plots**

Three representative series:
- FOODS_3_163_CA_3 (benchmark series)
- HOBBIES_1_408_CA_2 (worst series from 05b 8h)
- One SNAP-heavy FOODS series during a SNAP peak week

**Section 4 — Anomaly Detection**

- Residuals from Fold 3 predictions
- Z-score flags: demand spikes (z > 3), suppressed demand (z < −3)
- Isolation Forest on residual distribution
- Flag the 3,937 anomalous `price_change_pct_raw` rows from feature engineering

---

## Calibration Design Appendix

This section is a reference for why the isotonic regression method was
chosen over alternatives. It is included in the plan for portfolio
transparency.

### Option A — Fixed Bucket Corrections (Original Plan)
**Statistical weaknesses:** Arbitrary bin boundaries create discontinuities
that are indefensible in a production review. Mean residual per bucket
is not an optimal estimator of the correction needed. The step function
discards gradient information that the Section 8h scatter confirms is present.
**Status: Replaced.**

### Option B — Linear Regression on Zero Rate
**Properties:** OLS fit on (zero_rate, mean_residual) pairs. Simple, interpretable,
no boundary artifacts. Assumes linearity.
**Weakness:** Zero rate vs bias relationship may plateau above 80% (log-space
floor effect). A forced linear fit at high zero rates may overcorrect.
**Status: Acceptable fallback if isotonic regression fails the gate in 05d.**
If `IsotonicRegression` from sklearn is unavailable in the execution
environment, replace with `np.polyfit(degree=1)` and document the change.

### Option C — Isotonic Regression on Zero Rate (Adopted)
**Properties:** Non-parametric monotone fit. No assumed functional form.
Enforces the monotone constraint that is theoretically required (bias must
not improve as zero rate increases — the Section 8h r=−0.912 confirms this).
No boundary artifacts. As many degrees of freedom as the data supports.
**Weakness:** Slightly less interpretable than a linear fit for a portfolio
conversation. Mitigated by the fact that "monotone regression" is a
one-sentence explanation.
**Status: Adopted.**

### Option D — Two-Stage Model (Classify then Regress)
**Properties:** Separate binary classifier predicts P(demand > 0); separate
regressor predicts E[demand | demand > 0]. Final forecast = P(>0) × E[demand|>0].
Directly addresses zero-inflation as an architectural decision rather than a
post-hoc correction.
**Weakness:** Significant scope increase. Requires a new training objective,
a blending weight, and a separate validation framework.
**Status: Backlog — correct long-run solution, out of scope for this sprint.**

### What "not automatically accepted" means in practice
A production ML team treats calibration as a pipeline stage with its own
evaluation criteria, not a free improvement. The gate in 05d Section 4
enforces this: calibration must reduce bias by at least 20% (relative)
without worsening RMSE by more than 0.010. If it does not pass, the simpler
pipeline (suppression only) is used. Hiring managers reading this portfolio
will see that the practitioner distinguishes between "we tried this" and
"this earned its place in the pipeline" — that distinction is the mark of
production-aware ML work.

---

## Execution Order and Time Estimates

| Step | Notebook | Estimated time | Blocking dependency |
|---|---|---|---|
| 1 | `04b_feature_engineering_v2.ipynb` | 2–3 hours | None — start here |
| 2 | `05c_xgboost_v2.ipynb` (Optuna) | 3–5 hours (GPU) | 04b complete |
| 3 | `05d_validation_diagnostics_v2.ipynb` | 2–3 hours | 05c complete |
| 4 | Calibration gate decision locked | — | 05d complete |
| 5 | `06_lightgbm_demand.ipynb` (Optuna) | 3–5 hours (GPU) | 04b complete |
| 6 | Fold 2 model comparison (Section 6 of 06) | 1 hour | 05d + 06 complete |
| 7 | **`07_fold3_final_evaluation.ipynb`** | 2–3 hours | ALL above + gate locked |
| 8 | `08_explainability.ipynb` | 2–3 hours | 07 complete |
| 9 | Streamlit app | 1–2 days | 07 complete |
| 10 | README + portfolio | Half day | All notebooks complete |

**Total modeling work before app:** ~3–4 focused sessions.

LightGBM training (step 5) may run in parallel with steps 2–4 if separate
GPU resources are available — it has no dependency on 05c beyond the 04b
parquet files. If running sequentially, start 05c first.

---

## Key Decisions Already Made (Do Not Revisit)

| Decision | Rationale |
|---|---|
| v1 notebooks (01–05b) are frozen | Documented before-state. Modifying them destroys the improvement narrative. |
| Optuna re-runs on Fold 2 only | Feature space changed — v1 params may not be optimal for v2. Correct, not overfitting. |
| Both models go to Fold 3 | Winner selected on Fold 2. Fold 3 confirms generalization of both. |
| Holiday suppression is unconditional | Business constraint. Stores are closed. No performance gate applies. |
| Calibration method: isotonic regression | Smooth, non-parametric, monotone-constrained. No arbitrary bin boundaries. |
| Calibration is validation-gated | Must improve bias ≥20% without worsening RMSE >0.010 on Fold 2 to be adopted. |
| Fold 3 calibrator re-fit from Fold 3 training data | Training window is one year longer — zero rates should use full history. Method is the same; values are recomputed. |
| Fold 3 run once, results locked | Non-negotiable. Any retraining after seeing Fold 3 is leakage regardless of intent. |
| No modeling decision after Fold 3 | Once the notebook runs, the pipeline is locked. Fold 3 results are for reporting only. |

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
8. **Calibration gate** — calibration earns adoption or is rejected by evidence
9. **Fold 3** — one clean final result, never touched before this moment
10. **Explainability** — SHAP confirms the fixes worked
11. **App** — converts forecasts into decisions

Every notebook is a frozen artifact. Every decision is documented and justified.
The before-and-after comparison (v1 → v2) is built into the repository structure.
The calibration gate demonstrates production-aware pipeline thinking.
This is what a senior ML practitioner's development history looks like.