# Forward Plan — Retail Demand Intelligence System
## Updated Strategy: SKU-Level Inventory Decision Engine

**Current state:** `06b_model_selection.ipynb` complete. Tweedie LightGBM is
the production model. Global isotonic calibration rejected — overcorrects
at series level (median demand ratio 2.3–2.5×, p90 at 7–8× actual demand).
Winner lock-in (`WINNER_MODEL`, `WINNER_VARIANT`) is the immediate next step.

**Fold 3 discipline:** Fold 3 is run exactly once, at the very end of this
plan, after all modeling, optimization, uncertainty quantification, and
inventory simulation are locked on Fold 2. Any decision to retrain after
seeing Fold 3 results is test set contamination. This is non-negotiable.

**Core narrative:** This project is not a forecasting model. It is a
SKU-level inventory decision engine. Every notebook from here forward
serves one question: given a forecast and its uncertainty, what is the
correct reorder decision for this SKU right now?

---

## What 06b Established (Locked — Do Not Revisit)

| Decision | Rationale |
|---|---|
| Production model: LightGBM Tweedie | Predicts in unit space directly, no retransformation bias, best demand ratio across most zero-rate buckets |
| Production variant: suppressed only | Holiday suppression unconditional. Isotonic calibration rejected — series-level overcorrection proved by demand ratio distribution |
| XGBoost retired as primary candidate | Systematic underprediction (demand ratio 0.761), log-space retransformation bias. Retained in comparison tables only |
| Primary metric: per-SKU weekly WAPE | Aggregate WAPE masks SKU-level behavior. All optimization decisions from here use SKU-level WAPE distribution, not aggregate |
| Calibration status: rejected | Median demand ratio 2.315 (XGB) / 2.516 (Tweedie) post-calibration. p90 at 7–8× actual. Architecturally wrong for series-level correction |

---

## Key Architectural Decisions for the Forward Plan

### Decision 1 — SKU Routing: Demand Regime Classification (Syntetos-Boylan)

**Problem:** A single global model cannot serve all SKU types well. Forcing
Tweedie on Lumpy/Intermittent SKUs produces WAPE > 100% with no path to
improvement — the signal is not there.

**Solution:** Route SKUs to the appropriate forecasting method based on their
demand pattern, not on model performance metrics. Demand regime is a property
of the historical sales data — it requires no ground truth and no model
output to compute.

**Method:** Syntetos-Boylan (2005) classification using two statistics
computed from the training window:

- **ADI (Average inter-demand interval):** Total periods ÷ number of nonzero
  periods. Measures how frequently demand occurs.
- **CV² (Coefficient of variation squared):** Variance ÷ mean² of nonzero
  demand observations only. Measures how variable demand size is when it does
  occur.

**Four regimes from the 2×2 grid:**

| | CV² < 0.49 | CV² ≥ 0.49 |
|---|---|---|
| **ADI < 1.32** | Smooth — regular, stable demand | Erratic — regular, volatile demand |
| **ADI ≥ 1.32** | Intermittent — rare, stable demand | Lumpy — rare, volatile demand |

**Routing:**
- Smooth + Erratic → LightGBM Tweedie (global model handles these well)
- Intermittent → Croston/TSB (separates demand frequency from demand size)
- Lumpy → TSB or historical min-max policy (model-free fallback)

**Why this generalizes to production without ground truth:**
ADI and CV² are recomputed from the expanding training window every retraining
cycle. The classification updates as a SKU's behavior changes. No future
demand is needed at any point. This is the same mechanism used in SAP IBP,
Oracle Demantra, and Blue Yonder.

**Regime drift handling:**
To prevent boundary flip-flopping, apply hysteresis — a SKU only reclassifies
if it stays in the new regime for 4+ consecutive weeks. SKUs near regime
boundaries get flagged in the monitoring dashboard.

### Decision 2 — Uncertainty Quantification: Conformal Prediction + Native Croston Variance

**Problem:** Quantile regression requires separate models per quantile (q80,
q90, q95, q99), has no coverage guarantee, and can produce crossing quantiles.
Per-SKU safety stock cannot be principled without calibrated intervals.

**Solution:** Wrap the Tweedie model with conformal prediction. This produces
guaranteed-coverage prediction intervals at any service level from a single
model.

**How it works:**
1. Tweedie model trained on Fold 2 training data as normal
2. Run on Fold 2 val set — compute per-SKU residuals
3. For each SKU, the empirical quantiles of its residual distribution define
   the prediction intervals: the 90th percentile residual IS the q90 interval
   by mathematical construction
4. At inference: point forecast ± calibrated interval per SKU per service level
5. Intervals automatically widen for uncertain SKUs and narrow for stable ones

**Coverage guarantee:** If the calibration set is representative of the
production distribution, empirical coverage matches the target level exactly.
No distributional assumptions required. The q90 band covers 90% of actuals.

**Service levels supported from one model:** q50, q75, q80, q90, q95, q99.
User selects service level in the app; intervals come from the same conformal
wrapper.

**For Croston/TSB SKUs:** Native uncertainty comes from the variance of the
demand size estimates and inter-demand interval estimates. No conformal
wrapper needed — the Croston output already provides a natural buffer range.

**Why overprediction is intentional:** Carrying cost < stockout cost in most
retail contexts. Using q75 or q80 as the reorder quantity rather than q50
deliberately biases toward slight overstock, which is the correct business
default. Service level is configurable per SKU in the app.

### Decision 3 — No Global Post-Hoc Corrections

Global isotonic calibration was rejected in 06b. Department-level corrections
are rejected for the same reason — they are the same architectural mistake at
a finer granularity. They fit to Fold 2 val residual patterns that may not
generalize and add pipeline complexity without addressing the root cause.

The correct response to systematic bias on a SKU segment is: route those SKUs
to a better-suited model (Croston/TSB), not correct a bad model's output.

---

## Immediate Next Step — Lock 06b

Before any new notebook is opened, fill in the winner decision block in 06b:

```python
WINNER_MODEL   = 'tweedie'
WINNER_VARIANT = 'suppressed'
RATIONALE      = (
    "Tweedie wins on unit-space RMSE (3.10 vs 3.39), demand ratio (0.776 vs "
    "0.761), and WAPE across 4 of 5 zero-rate buckets. Predicts in unit space "
    "directly — no retransformation bias. Global isotonic calibration rejected: "
    "series-level demand ratio explodes to 2.3–2.5× median, p90 at 7–8× actual. "
    "XGBoost retired. Holiday suppression unconditional."
)
```

Save `winner_decision_fold2.pkl`. This is the formal gate. No new notebook
runs until this is committed.

---

## Repository Structure (Updated)

```
retail-demand-intelligence/
├── data/
│   ├── raw/
│   └── processed/
│       ├── features/
│       │   ├── features_train_v2.parquet
│       │   ├── features_val_v2.parquet
│       │   ├── feature_cols_v2.pkl
│       │   └── item_mean_price_lookup.pkl
│       ├── models/
│       │   ├── xgb_v2_model_fold2.json          ← frozen, comparison only
│       │   ├── tweedie_model_fold2.txt           ← production candidate
│       │   ├── tweedie_optimized_fold2.txt       ← 06d output
│       │   └── tweedie_optimized_fold3.txt       ← 07 output
│       ├── predictions/
│       │   ├── tweedie_predictions_fold2.parquet
│       │   ├── tweedie_optimized_predictions_fold2.parquet
│       │   ├── croston_predictions_fold2.parquet
│       │   ├── final_predictions_fold2.parquet   ← routed, all SKUs
│       │   └── final_predictions_fold3.parquet   ← 07 output
│       ├── calibration/
│       │   ├── winner_decision_fold2.pkl         ← 06b lock-in
│       │   ├── tweedie_best_params.pkl
│       │   ├── conformal_residuals_fold2.pkl     ← 06e output
│       │   └── conformal_residuals_fold3.pkl     ← 07 output
│       ├── segmentation/
│       │   ├── sku_regimes_fold2.parquet         ← 06c output
│       │   ├── sku_regimes_fold3.parquet         ← 07 output
│       │   └── regime_thresholds.pkl
│       └── inventory/
│           ├── reorder_parameters_fold2.parquet  ← 06g output
│           ├── simulation_results_fold2.parquet  ← 06g output
│           └── simulation_results_fold3.parquet  ← 07 output
├── notebooks/
│   ├── 01_eda.ipynb                              ✅ frozen
│   ├── 02_baselines_and_stats.ipynb              ✅ frozen
│   ├── 03_prophet.ipynb                          ✅ frozen
│   ├── 03b_prophet_stores_conclusion.ipynb       ✅ frozen
│   ├── 04_feature_engineering.ipynb              ✅ frozen
│   ├── 04b_feature_engineering_v2.ipynb          ✅ frozen
│   ├── 05_xgboost_demand.ipynb                   ✅ frozen
│   ├── 05b_validation_diagnostics.ipynb          ✅ frozen
│   ├── 05c_xgboost_v2.ipynb                      ✅ frozen
│   ├── 05d_validation_diagnostics_v2.ipynb       ✅ frozen
│   ├── 06_lightgbm_demand.ipynb                  ✅ frozen
│   ├── 06b_model_selection.ipynb                 ← finish now (lock winner)
│   ├── 06c_sku_audit.ipynb                       ← Phase A
│   ├── 06d_model_optimization.ipynb              ← Phase B
│   ├── 06e_uncertainty_quantification.ipynb      ← Phase C
│   ├── 06f_intermittent_demand.ipynb             ← Phase D
│   ├── 06g_inventory_simulation.ipynb            ← Phase E
│   ├── 07_fold3_final_evaluation.ipynb           ← Phase F (run once)
│   ├── 08_explainability.ipynb                   ← Phase G
│   └── 09_app_data_prep.ipynb                    ← Phase H
├── app.py
├── requirements.txt
└── README.md
```

**Rule:** Notebooks 01–06b are frozen artifacts. They document the full
development history and are never modified. The improvement narrative
(v1 → v2 → Tweedie → optimized → inventory engine) is built into the
repository structure itself.

---

## Phase A — SKU Failure Audit and Demand Regime Classification
### Notebook: `06c_sku_audit.ipynb`

**Purpose:** Understand exactly where and why the model fails before
touching anything. Classify every SKU by demand regime. This is diagnostic
only — no modeling, no optimization.

**Section 1 — Per-SKU Weekly WAPE Audit**

Compute per-SKU weekly WAPE from the 06b predictions (Tweedie suppressed).
This is the primary diagnostic — not aggregate WAPE.

- Full histogram of per-SKU weekly WAPE (not percentiles — the full shape)
- WAPE vs mean daily demand scatter — is catastrophic error concentrated in
  low-volume SKUs?
- WAPE vs zero rate scatter — find the cliff where the model breaks
- Per-SKU demand ratio distribution — what does the tail look like on both
  sides?
- What % of total sales revenue sits in each WAPE band — a SKU with 200%
  WAPE matters far less if it represents 0.1% of revenue

Do not define tiers from WAPE. This section is purely diagnostic. The tiers
come from the demand regime classification below.

**Section 2 — Demand Regime Classification (Syntetos-Boylan)**

Compute ADI and CV² for every SKU from the Fold 2 training window
(Feb 2011 → Jan 2014). These are properties of the historical sales data,
not of model performance.

```python
def compute_adi(series: pd.Series) -> float:
    """Average inter-demand interval. Total periods / nonzero periods."""
    n_total   = len(series)
    n_nonzero = (series > 0).sum()
    return n_total / n_nonzero if n_nonzero > 0 else np.inf

def compute_cv2(series: pd.Series) -> float:
    """CV squared of nonzero demand observations only."""
    nonzero = series[series > 0]
    if len(nonzero) < 2:
        return np.inf
    return (nonzero.std() / nonzero.mean()) ** 2

def classify_regime(adi: float, cv2: float) -> str:
    """Syntetos-Boylan (2005) classification."""
    if adi < 1.32 and cv2 < 0.49:
        return 'smooth'
    elif adi < 1.32 and cv2 >= 0.49:
        return 'erratic'
    elif adi >= 1.32 and cv2 < 0.49:
        return 'intermittent'
    else:
        return 'lumpy'
```

Thresholds: ADI = 1.32, CV² = 0.49 (Syntetos & Boylan, 2005 — cite this).

**Section 3 — Regime Distribution and Cross-Validation**

- How many SKUs in each regime? What % of total revenue per regime?
- Cross-tabulate regime vs zero-rate bucket from 06b — confirm they align
- Cross-tabulate regime vs per-SKU WAPE from Section 1 — confirm Smooth
  SKUs have lower WAPE, Lumpy SKUs have higher WAPE. If this relationship
  does not hold, investigate before proceeding.
- Show 3 representative SKUs per regime with their demand pattern plots

**Section 4 — Routing Assignment**

```python
def assign_routing(regime: str) -> str:
    routing = {
        'smooth':       'tweedie',
        'erratic':      'tweedie',
        'intermittent': 'croston',
        'lumpy':        'tsb_or_policy',
    }
    return routing[regime]
```

**Section 5 — Hysteresis Gate Design**

Document the hysteresis rule: a SKU reclassifies only if it stays in a new
regime for 4+ consecutive weekly recomputation cycles. SKUs within 10% of
either threshold boundary are flagged as "boundary SKUs" in the output.

**Outputs:**
- `sku_regimes_fold2.parquet` — SKU, ADI, CV², regime, routing, boundary_flag
- `regime_thresholds.pkl` — threshold values (1.32, 0.49) for reuse at Fold 3

---

## Phase B — Model Optimization
### Notebook: `06d_model_optimization.ipynb`

**Purpose:** Improve Tweedie point forecast accuracy on Smooth and Erratic
SKUs by testing three targeted architectural changes against a fair baseline.
Do not attempt to fix Intermittent or Lumpy SKUs here — they are handled
in Phase D.

**Core principle:** Hyperparameters are frozen from 06b. Each experiment
changes exactly one architectural decision — target variable or loss
function. This is a controlled comparison, not a new hyperparameter search.
Changing one thing at a time means if an experiment wins, we know why.

**Evaluation metric:** Per-SKU weekly WAPE distribution on Smooth + Erratic
SKUs only. An experiment is adopted only if it improves median per-SKU
weekly WAPE by ≥ 5% without worsening p90. If no experiment clears this
bar, the 06b model is retained and documented as the production model.
A null result is a valid and honest finding.

---

**Experiment 0 — Fair Baseline**

Re-evaluate 06b Tweedie Raw filtered to Smooth + Erratic SKUs only.
Prior evaluations included Intermittent and Lumpy SKUs which suppressed
the headline numbers. The true performance on forecastable SKUs is the
correct baseline for all comparisons.

Metrics to document: median weekly WAPE, p75, p90, % SKUs < 30%,
% SKUs < 50%, % SKUs > 100%, median demand ratio, % SKUs with demand
ratio in 0.8–1.2 band.

---

**Experiment A — Direct 7-Day Target**

Retrain Tweedie with frozen 06b hyperparameters, changing only the target
from next-day demand to 7-day forward sum.

Rationale: the model currently trains on daily demand but inventory
decisions are made weekly. This horizon mismatch means the model is
optimizing the wrong signal. Training directly on the 7-day sum eliminates
daily error accumulation and aligns the optimization target with the
actual decision being made.

Implementation: for each row in the training set, compute
`target_7d = sum of units_sold over the next 7 calendar days`.
Rows within 7 days of the training window end are dropped — no lookahead.
Everything else identical to 06b.

No Optuna. Frozen params. One retrain.

---

**Experiment B — Asymmetric Loss**

Retrain Tweedie with frozen 06b hyperparameters and daily target,
replacing the Tweedie loss with a custom asymmetric objective.

Rationale: stockout cost > holding cost in retail. A symmetric loss
function treats a 10-unit underforecast identically to a 10-unit
overforecast. The asymmetric loss directly encodes the business reality
that underforecasting is more expensive.

Alpha controls the asymmetry — the single parameter being searched:

```python
alpha = 0.50  # symmetric — identical to standard loss
alpha = 0.60  # underforecast penalized 1.5× more
alpha = 0.70  # underforecast penalized 2.3× more  
alpha = 0.80  # underforecast penalized 4.0× more
```

Alpha grid: 16 evenly spaced values from 0.50 to 0.80. For each value,
retrain once with frozen params and evaluate median demand ratio on val.
Optimal alpha = value that puts median demand ratio closest to 1
(slight upward bias appropriate for retail) without pushing p90 above 1.4.
This is a simple loop, not Optuna — one parameter, smooth landscape,
no intelligent sampling needed.

---

**Experiment C — A + B Combined**

Retrain with frozen params, 7-day target, and best alpha from Experiment B.
Tests whether the two improvements compound. If both A and B individually
help, C is likely the winner. If only one helps, C may or may not beat
the individual winner — let the data decide.

---

**Section 5 — Comparison and Winner Selection**

Full distribution comparison across Experiments 0, A, B, C:

| Metric | Exp 0 | Exp A | Exp B | Exp C |
|---|---|---|---|---|
| Median WAPE | | | | |
| p75 WAPE | | | | |
| p90 WAPE | | | | |
| % SKUs < 30% | | | | |
| % SKUs > 100% | | | | |
| Median demand ratio | | | | |
| % ratio in 0.8–1.2 | | | | |

Winner selected on median per-SKU weekly WAPE. Adopted only if ≥ 5%
improvement over Experiment 0. Otherwise 06b model retained as-is.

**Outputs:**
- `tweedie_optimized_fold2.txt` — winning model (may be 06b model unchanged)
- `tweedie_optimization_results.pkl` — full comparison table for portfolio
---

## Phase C — Uncertainty Quantification
### Notebook: `06e_uncertainty_quantification.ipynb`

**Purpose:** Generate calibrated prediction intervals for every Smooth and
Erratic SKU at multiple service levels. These intervals directly drive
safety stock calculations in Phase E.

**Section 1 — Conformal Prediction Setup**

Conformal prediction wraps the optimized Tweedie model and produces
coverage-guaranteed intervals at any service level from a single calibration
step.

**How it works:**

```python
# Step 1: Generate predictions on Fold 2 val set
# (training is already done — this is post-hoc calibration)
val_preds = optimized_tweedie.predict(X_val_smooth_erratic)
val_actuals = y_val_smooth_erratic

# Step 2: Compute per-SKU residuals on val set
residuals_df = pd.DataFrame({
    'id':       val['id'].values,
    'residual': val_actuals - val_preds   # actual - predicted
})

# Step 3: For each SKU, store the empirical residual distribution
# These ARE the prediction intervals by construction
sku_residuals = residuals_df.groupby('id')['residual'].apply(list)

# Step 4: At inference, for a new prediction:
# Lower bound at service level alpha = point_forecast + quantile(residuals, 1-alpha)
# Upper bound = point_forecast + quantile(residuals, alpha)
# For inventory we primarily care about the upper bound:
# reorder_upper_q90 = point_forecast + np.percentile(sku_residuals[sku_id], 90)
```

**Why this is valid:** The conformal guarantee holds as long as the
calibration residuals are exchangeable with the production residuals — i.e.,
that Fold 2 val demand patterns are representative of future demand patterns.
This is the same assumption the model itself makes. No additional assumptions
are required.

**Section 2 — Coverage Validation**

This is the critical validation step. For each service level, verify that
empirical coverage matches the target on the Fold 2 val set.

```
Coverage Validation — Fold 2 Val (Smooth + Erratic SKUs):

Service level  Target coverage  Empirical coverage  Pass/Fail
q50            50%              [result]%           [P/F]
q75            75%              [result]%           [P/F]
q80            80%              [result]%           [P/F]
q90            90%              [result]%           [P/F]
q95            95%              [result]%           [P/F]
q99            99%              [result]%           [P/F]

Pass threshold: within ±3 percentage points of target.
```

If coverage fails at any level, investigate whether it is a global bias
(all SKUs off) or concentrated in specific regimes. Do not proceed to
Phase E until coverage passes.

**Section 3 — Per-SKU Interval Width Analysis**

- Distribution of interval widths at q90 across all Smooth + Erratic SKUs
- Interval width vs zero rate scatter — confirm intervals widen for sparser SKUs
- Interval width vs mean demand scatter — confirm intervals are proportional
- Flag SKUs where q90 interval width > 3× mean demand (extreme uncertainty —
  these are borderline Intermittent and may need rerouting)

**Section 4 — Service Level Selection by Regime**

Document the recommended default service level per regime:

| Regime | Default service level | Rationale |
|---|---|---|
| Smooth | q80 | Low forecast error, moderate buffer sufficient |
| Erratic | q90 | High demand variance, wider buffer needed |
| Intermittent | Croston native | No conformal — see Phase D |
| Lumpy | Policy-based | No conformal — see Phase D |

These are defaults. The app allows override per SKU.

**Outputs:**
- `conformal_residuals_fold2.pkl` — per-SKU residual distributions
- `coverage_validation_fold2.csv` — coverage table for portfolio

---

# Phase D — Intermittent Demand Handling

### Notebook: `06f_intermittent_demand.ipynb`

**Purpose:** Provide principled forecasts for Intermittent SKUs via TSB, implement policy-based fallback for Lumpy SKUs, and empirically validate that domain-specific methods outperform the global Tweedie model on these regimes.

---

## Section 1 — Setup & Regime Isolation

Load 06c outputs and isolate SKUs by demand regime:

* **Intermittent:** 14,268 SKUs (`ADI ≥ 1.32`, `CV² < 0.49`) → TSB method
* **Lumpy:** 7,003 SKUs (`ADI ≥ 1.32`, `CV² ≥ 0.49`) → Historical Policy
* **Smooth + Erratic:** 9,219 SKUs → Already handled by 06d/06e

Confirm ID match and validate zero-fraction distribution.

---

## Section 2 — TSB Implementation for Intermittent SKUs

Implement **TSB (Teunter-Syntetos-Babai)** with probability decay:

* **p:** Demand probability, updated every period, including zeros
* **z:** Non-zero demand size, updated only when demand occurs
* **Forecast:** `p × z` — expected demand per period

**Why TSB:** Prevents indefinite obsolete-SKU forecasting, addressing the known Croston/SBA failure mode.

**Validation:** Perform an in-sample mechanism check across all 14,268 Intermittent SKUs.

---

## Section 3 — Parameter Selection (Walk-Forward, Fold 2 Split)

Grid search:

* `α, β ∈ {0.05, 0.1, 0.2, 0.3}`
* Use a 5,000-SKU sample for parameter selection
* Evaluate on held-out Fold 2 validation weeks (`814 days`)
* Rank configurations by **median per-SKU weekly WAPE**
* Validate the selected parameters on the full population of 14,268 Intermittent SKUs

---

## Section 4 — Historical Policy for Lumpy SKUs

Implement a **min-max inventory policy** for the 7,003 Lumpy SKUs:

```python
def historical_policy_forecast(
    train_demand,
    lead_time_weeks=1,
    buffer_multiplier=1.25
):
    weekly_demand = train_demand.resample('W').sum()
    max_weekly = weekly_demand.max()
    avg_weekly = weekly_demand.mean()

    reorder_point = (
        max_weekly
        * lead_time_weeks
        * buffer_multiplier
    )

    order_qty = max(
        max_weekly * buffer_multiplier - avg_weekly,
        avg_weekly
    )

    return {
        'reorder_point': reorder_point,
        'order_qty': order_qty,
        'max_weekly': max_weekly,
        'avg_weekly': avg_weekly
    }
```

**Why policy:** Lumpy SKUs have insufficient signal for reliable point forecasting.

**Buffer:** Default = `1.25` (retail standard, approximately 80–85% service level). Tune in 06g.

---

## Section 5 — Evaluation: TSB vs Tweedie on Intermittent SKUs

**Key portfolio validation:** Directly compare TSB against the global Tweedie model on the Fold 2 validation set.

### Metrics

* Per-SKU weekly WAPE
* Demand ratio distribution
* Improvement rate

### Target

**TSB >90% improvement rate over Tweedie on Intermittent SKUs.**

### Proof Objective

Demonstrate empirically that the **domain-specific intermittent-demand method outperforms the global ML model on sparse demand regimes**.

This section is critical because it provides the evidence supporting the routing decision rather than simply assuming that TSB is better for Intermittent SKUs.

---

## Section 6 — Unified Prediction Output

Combine all routing methods into a single prediction dataframe.

### Schema

```text
id | regime | routing | point_forecast | q50 | q75 | q80 | q90 | q95 | q99 | interval_source
```

### Intermittent — TSB

* `routing = TSB`
* `point_forecast = TSB forecast`
* `q50 = point_forecast`
* `q90 = point_forecast + 2 × std(nonzero demand)`

### Lumpy — Historical Policy

* `routing = historical_policy`
* `point_forecast = reorder_point`
* All quantiles equal `reorder_point`

### Outputs

* `croston_predictions_fold2.parquet`
* `final_predictions_fold2.parquet`
* `lumpy_policy_params.pkl`

---

# Structure Summary

| Section | Content                       | Status                                                  |
| ------- | ----------------------------- | -------------------------------------------------------- |
| 1       | Setup & Regime Isolation      | ✅ Done                                                   |
| 2       | TSB Implementation            | ✅ Done                                                   |
| 3       | Parameter Selection           | ✅ Done — α=β=0.5, 52% WAPE improvement over naive        |
| 4       | Historical Policy (Lumpy)     | ✅ Done — min-max policy, tiered buffer, order_qty floors |
| 5       | **TSB vs Tweedie Comparison** | ✅ Done — TSB beats Tweedie on 99.0% of matched SKUs      |
| 6       | Safety Stock (Intermittent)   | ✅ Done — see schema deviation note below                 |
| 7       | Unified Output (Lumpy+Intermittent) | ✅ Done — order_qty gap closed, see Pre-flight note in Phase E |

> **Section 5 (TSB vs Tweedie) is done and stronger than the original bar.** Median WAPE
> 31.76% (TSB) vs 50.09% (Tweedie), 99.0% win rate. The more important result for the
> portfolio narrative isn't the average — it's the tail: 0.0% of TSB forecasts exceed 100%
> WAPE vs. 13.3% for Tweedie, and Tweedie's worst SKU hits 2,202% WAPE vs. 114% for TSB.
> Tweedie doesn't just do worse on average on these SKUs, it periodically fails
> catastrophically — exactly the failure mode the regime routing exists to prevent.

> **Section 6's output schema deviates from the original spec above, deliberately.** The
> original design called for a shared `q50...q99` quantile schema across all routing methods
> (see the original Section 6 spec below, kept for reference). In practice, TSB's Bernoulli(p)
> × bootstrapped-size design under-covered by ~11pp in walk-forward validation (real demand
> clusters in time; independent-per-day resampling understates 7-day variance). The fix — a
> moving-block bootstrap that resamples real historical 7-day windows directly — produces a
> `safety_stock_q80` at one locked service level per SKU, not a full quantile ladder. This is
> the right tradeoff (82.1% empirically validated coverage vs. an unvalidated quantile schema)
> but it means Intermittent/Lumpy's actual output is `reorder_point`/`safety_stock`/`order_qty`
> directly, not `q50...q99`. **This has real consequences for 06g — see the Pre-flight section
> below before writing 06g Section 1.**

*(Original Section 6 spec, superseded by the above — kept for context, not a target to
re-implement):*

```text
id | regime | routing | point_forecast | q50 | q75 | q80 | q90 | q95 | q99 | interval_source
```

---

## Phase E — Inventory Simulation
### Notebook: `06g_inventory_simulation.ipynb`

**Purpose:** Convert forecasts → reorder decisions → simulate business
outcomes on the Fold 2 val window. Produce the headline business metrics
that anchor the portfolio narrative.

---

### Pre-flight — Reconciling This Plan With What 06f Actually Shipped

Written after 06f closed out. Four adaptations are required before Section 1 below can run
as originally written — none of them are optional, and none require redoing 06f.

**1. Only Smooth/Erratic needs `compute_reorder_params`. Lumpy and Intermittent already have
final numbers.** The function below (quantile-width → safety-stock) was designed as one
generic path for every regime. That assumption no longer holds:
- **Lumpy** (`unified_lumpy_intermittent_fold2.parquet`, `regime == 'lumpy'`): `reorder_point`,
  `order_qty` already computed via the min-max policy. Use directly.
- **Intermittent** (same file, `regime == 'intermittent'`): `reorder_point` (=
  `safety_stock_q80`) and `order_qty` (= TSB's `z_final`, floored) already computed via block
  bootstrap. Use directly.
- **Smooth/Erratic** (9,219 SKUs): this is the *only* regime that still needs
  `compute_reorder_params`. 06e never built a reorder table — it saved raw per-SKU residuals
  only (`conformal_residuals_fold2.pkl`: `sku_residuals`, `default_service_level=0.80`,
  `residual_unit`). Section 1 below needs to load that pickle plus
  `tweedie_optimized_predictions_fold2.parquet` (for `yhat`) and build `reorder_point`/
  `safety_stock` from scratch — this is genuinely new work, not a merge.

**2. `order_qty` for Smooth/Erratic is still undefined.** The original plan never specified a
replenishment quantity for the conformal-wrapped regime — only `reorder_point`. 06f resolved
the equivalent gap for Intermittent by using TSB's `z_final` (mean nonzero demand size) as the
cycle-stock unit. The Smooth/Erratic analogue is the model's own point forecast — e.g.
`order_qty = point_forecast_weekly × lead_time_weeks` (order one lead-time's worth of expected
demand) — but this needs an explicit decision here, the same way it did in 06f, not a silent
default inside `simulate_inventory`.

**3. `reorder_qty` in `simulate_inventory` below is a fixed value per SKU, decided once — not
computed dynamically inside the simulation loop.** The original docstring said
`reorder_qty = reorder_point - current_inventory (when triggered)` (an order-up-to-S policy),
but the function signature took a static `reorder_qty` argument — those two are inconsistent
policies and only one can be implemented. **Resolved: use the static, precomputed `order_qty`
column** (matches what 06f now produces for Lumpy/Intermittent, and what item 2 above proposes
for Smooth/Erratic) — an order-up-to-S policy would need `simulate_inventory`'s signature
changed to drop `reorder_qty` and compute it inline from `reorder_point`, which is a bigger
change and wasn't validated anywhere in this pipeline. Keep the fixed-quantity policy for
consistency across all three regimes; the code below reflects this.

**4. `sqrt(lead_time)` scaling in `compute_reorder_params` is an untested independence
assumption — validate it, don't inherit it.** 06e confirmed the winning model (`experiment_0`)
predicts in **next-day units**, so this scaling assumes i.i.d. daily forecast errors. 06f
Section 6 already found the opposite for Intermittent SKUs — real demand clusters in time,
and an i.i.d.-per-day simulation under-covered by ~11pp until replaced with a block bootstrap.
Smooth/Erratic demand is more regular and may hold up better under the independence
assumption, but "may" isn't a validated claim. **Before trusting Smooth/Erratic reorder
points, run the same rolling-origin coverage check 06f Section 6 used** (aggregate actual
7-day-forward demand vs. the `sqrt(7)`-scaled interval, walk-forward through the val period) —
reuse the pattern, don't assume it transfers.

**5. Reconcile cadence across all three regimes before the final merge, the same way 06f
Section 7 did for Lumpy vs. Intermittent.** Lumpy uses calendar-week (`resample('W')`)
aggregation; Intermittent uses a rolling daily block bootstrap over a 7-day window; Smooth/
Erratic will aggregate native next-day residuals into a 7-day figure in Section 1 below. Add
an explicit assert/cadence-check cell here (mirroring 06f Section 7's `LEAD_TIME_DAYS == 7`
assert) before concatenating all three into one table — don't assume alignment.

**6. `safety_buffer < 0` is informative, not an error, for a known Intermittent cohort.** 06f
Section 7 found that some Intermittent SKUs legitimately have `safety_stock_q80 = 0` with
positive expected demand (reactive-only reordering — no proactive buffer justified at that
SKU's sparsity, but `order_qty > 0` guarantees it still restocks on depletion). Carry the
`regime` + `safety_buffer < 0` combination into Section 5's per-regime breakout rather than
treating it as a data-quality flag.

---

**Section 1 — Reorder Parameter Computation**

For Smooth/Erratic SKUs only (see Pre-flight #1) — Lumpy and Intermittent read their
`reorder_point`/`order_qty` directly from `unified_lumpy_intermittent_fold2.parquet`:

```python
def compute_reorder_params(sku_id: str,
                            point_forecast_weekly: float,
                            q90_upper: float,
                            lead_time_weeks: int,
                            service_level: float) -> dict:
    """
    Compute reorder point, safety stock, and order quantity for a single
    Smooth/Erratic SKU from its conformal residual quantile.

    safety_stock  = (q_upper - point_forecast) × sqrt(lead_time)
                  = interval half-width scaled by lead time uncertainty.
                  NOTE: assumes i.i.d. daily forecast errors -- validate with a
                  rolling-origin coverage check (Pre-flight #4) before trusting this,
                  the same way 06f Section 6 validated the block-bootstrap alternative.
    reorder_point = point_forecast × lead_time + safety_stock
    order_qty     = point_forecast × lead_time (fixed, precomputed -- see Pre-flight #3)
    """
    interval_width = q90_upper - point_forecast_weekly
    safety_stock   = interval_width * np.sqrt(lead_time_weeks)
    reorder_point  = point_forecast_weekly * lead_time_weeks + safety_stock
    order_qty      = point_forecast_weekly * lead_time_weeks
    return {
        'sku_id':         sku_id,
        'point_forecast': point_forecast_weekly,
        'safety_stock':   safety_stock,
        'reorder_point':  reorder_point,
        'order_qty':      order_qty,
        'service_level':  service_level,
    }
```

**Section 2 — Inventory Depletion Simulation**

Simulate week-by-week inventory depletion on the Fold 2 val window
(Feb 2014 → Jan 2015) using actual sales as ground truth. `reorder_qty` is a fixed,
precomputed value per SKU (Pre-flight #3) — sourced from `order_qty` in the unified table for
all three regimes, not recomputed inside the loop.

```python
def simulate_inventory(actual_weekly_demand: np.ndarray,
                        reorder_point: float,
                        reorder_qty: float,
                        initial_inventory: float,
                        lead_time_weeks: int) -> dict:
    """
    Simulate inventory trajectory under a fixed-quantity reorder policy.
    Returns stockout weeks, average inventory, service level achieved.
    """
    inventory     = initial_inventory
    stockout_weeks = 0
    inventory_history = []
    pending_order = 0
    weeks_until_arrival = 0

    for week, demand in enumerate(actual_weekly_demand):
        # Receive order if due
        if weeks_until_arrival == 0 and pending_order > 0:
            inventory += pending_order
            pending_order = 0

        # Fulfill demand
        fulfilled = min(inventory, demand)
        if demand > inventory:
            stockout_weeks += 1
        inventory -= fulfilled

        # Place reorder if below reorder point
        if inventory <= reorder_point and pending_order == 0:
            pending_order       = reorder_qty
            weeks_until_arrival = lead_time_weeks

        inventory_history.append(inventory)
        if weeks_until_arrival > 0:
            weeks_until_arrival -= 1

    return {
        'stockout_rate':     stockout_weeks / len(actual_weekly_demand),
        'avg_inventory':     np.mean(inventory_history),
        'fill_rate':         1 - stockout_weeks / len(actual_weekly_demand),
        'inventory_history': inventory_history,
    }
```

**Section 3 — Simulate at All Service Levels**

Run the simulation at q50, q75, q80, q90, q95, **and q99** for all SKUs (extended from the
original q50-q95 range once Section 4's cost analysis showed the tested range wasn't wide
enough to bracket a cost minimum -- see Section 4 below). Track `units_short` (cumulative
unfulfilled demand per SKU) alongside `stockout_rate`, `avg_inventory`, and `fill_rate` --
required by Section 4's cost formula; a stockout-week *count* is not the same as unit
shortfall and the two are not proportional across regimes with different order sizes.
Aggregate results by regime and service level, and pool across all SKUs into one headline
table -- **this pooled table is the headline result of the entire project.**

**Section 4 — Cost Sensitivity Analysis**

**Revised from the original plan.** The original spec asked for one cost scenario (fixed
$4.00 unit cost, 2x stockout multiplier, 25% carrying rate) and a single crossover point.
That approach was abandoned after the locked-scenario run showed cost decreasing
monotonically through q99 with no interior minimum -- reporting a crossover point that
wasn't actually there would have repeated the exact class of error Section 3 caught
earlier (pricing something that wasn't validated). Two changes were made instead:

1. **Real per-SKU unit cost, not a flat $4.00.** `sell_prices.csv` has actual weekly
   retail sell_price per item_id/store_id, already used elsewhere in this pipeline
   (01_eda, 04_feature_engineering, 05b) -- joined and averaged over the Fold 2 val window
   per SKU, with portfolio-median fallback for SKUs missing a price row. Caveat carried
   into every downstream cost figure: this is retail sell_price, not wholesale/COGS --
   M5 has no cost field, so this is the same proxy the flat $4.00 was already
   approximating, just no longer pooled into one number across 30,442 heterogeneous SKUs.

2. **Sensitivity sweep instead of one locked scenario.** Stockout multiplier swept
   {1.5x, 2.0x, 3.0x, 4.0x, 6.0x, 8.0x} x unit cost, carrying rate swept
   {15%, 20%, 25%, 30%} annual -- 24 combinations, each re-costing the same six
   already-validated service levels plus naive. Report where the cost-minimizing
   service level moves (or doesn't) across the full grid, not just at the plan's
   original 2.0x/25% default.

**Cross-check against theory:** the newsvendor critical ratio
`SL* = stockout_cost / (stockout_cost + carrying_cost)` gives an independent prediction
for the cost-optimal service level at each tested ratio. Compare the sweep's empirical
optimum against this theoretical value at each of the 24 combinations -- if the sweep's
tested range (q50-q99) doesn't reach the theoretical optimum, say so explicitly rather
than reporting the boundary of the tested range as if it were the true minimum.

**Known limitation, named rather than fixed here:** carrying cost is modeled as strictly
linear in avg_inventory (`avg_inventory x unit_cost x carry_rate`). Real overstock cost is
convex, not linear -- shelf/warehouse capacity constraints, markdown/obsolescence risk
(relevant given FOODS categories in this dataset), and fixed working-capital budgets all
put a real ceiling on "more inventory is better" that this model does not capture. Flagged
as explicit future work, not solved in this pass: (a) convex or capacity-constrained
carrying cost, (b) a category-specific obsolescence term for perishables, (c) per-SKU
newsvendor-optimal service levels (since `Cu`/`Co` is SKU-specific) instead of one blanket
service level swept across the whole portfolio.

Report the dollar figure at q80 vs. naive baseline as the headline business number for the
README, **stated with its scope**: "under real per-SKU retail pricing, at q80 service
level, the system reduces total inventory cost by approximately $X vs. naive reorder,
and this advantage is stable (range: Y%-Z%) across 24 plausible retail cost-ratio
assumptions." Do not claim q99 (or any tested boundary point) as the confirmed cost
optimum -- report it only as the best point within the validated service-level range.

**Section 5 — Simulation by SKU Regime**

Break out simulation results by regime (Smooth, Erratic, Intermittent, Lumpy) across all
six tested service levels. Confirm, with appropriate nuance rather than a binary yes/no:
- Whether Smooth SKUs achieve comparatively high fill rates, and at which service levels
  specifically -- check for crossovers against Erratic rather than assuming Smooth leads
  uniformly.
- Whether Lumpy's fixed min-max policy fill rate falls within the buffer parameter's
  documented expected range (Phase E setup: 1.25x buffer, ~80-85% service level) --
  Lumpy has no quantile lever in this sweep, so "requires high service levels" should be
  read as "requires an adequately-tuned fixed buffer," not literally interpreted as a
  quantile claim.

**Additionally, once Section 4's real per-SKU pricing exists:** break out cost
contribution by regime at q80 (stockout cost, carrying cost, total cost) alongside each
regime's SKU-count share. Report cost share **separately for carrying cost**, not just
total cost -- stockout cost dominates total cost heavily enough that a regime can be
badly over-represented in carrying cost specifically while looking proportional at the
total-cost level. This is the diagnostic that identifies which regime's buffer/policy
parameter is worth revisiting if carrying cost (rather than total cost) becomes a target.

**Outputs:**
- `final_reorder_params_fold2.parquet` (Section 1) -- filename as actually produced;
  earlier plan draft named this `reorder_parameters_fold2.parquet` without the `final_`
  prefix, corrected here to match what's on disk.
- `simulation_results_fold2.parquet` (Section 2)
- `service_level_sweep_fold2.parquet` (Section 3)
- `naive_baseline_fold2.parquet` (Section 3)
- `cost_sensitivity_fold2.parquet` (Section 4)
- Headline numbers locked for README and app, stated with cost-assumption scope per
  Section 4 above


  ### Post-Section-6 Addendum — Periodic-Review Policy Promotion (before Phase F)

Section 6's diagnostic surfaced that Sections 1–5's production policy is continuous-review
(s,Q) — reorder point and quantity set once, held fixed for the full val window. This does
not match the weekly-cadence replenishment pattern the project is meant to model, and the
forecast (the system's actual product) is used once instead of every review cycle.

**Bug found and fixed:** the periodic-review order-up-to target omitted the review-period
term (`protection_weeks = lead_time` instead of `lead_time + review_period`), and initial
dynamic-policy runs reused static's lead-time-only safety stock unmodified — both bias the
policy toward appearing to need more inventory than it should, and toward looking artificially
strong (never stocking out) without being genuinely calibrated. Fixed: dynamic policy gets its
own safety stock, independently swept at q50–q99 for a `sqrt(lead_time + review_period)`
window, not borrowed from static.

**Decision: Option C.** Promote periodic-review to production; retain static (s,Q) as the
textbook-baseline comparator (same role Naive already plays relative to static). Three-tier
story: Naive → Static (s,Q) → Dynamic (periodic-review).

**New sections 7–9 added to 06g before Phase F:**
- Section 7 — Dynamic policy calibration at full population (all ~30,442 SKUs, not the
  12-SKU sample). Real weekly forecast per regime: Tweedie trajectory (Smooth/Erratic,
  reused from Section 6), TSB's native per-timestep trajectory (Intermittent — previously
  computed in 06f and discarded, now reused; removes the trailing-average proxy caveat
  entirely), fixed target (Lumpy — no model to refresh, by design).
- Section 8 — Dynamic cost sensitivity, same real-pricing/24-combination grid as Section 4.
- Section 9 — Head-to-head: static vs. dynamic, each at its own cost-minimizing service
  level. This supersedes Sections 3–5's headline numbers; static's tables are kept as the
  baseline row, not deleted.

**Re-validation required before trusting new numbers:** Pre-flight #4's rolling-origin
coverage check (originally validated `sqrt(lead_time)` scaling for a 1-week window) must be
rerun at the widened `sqrt(lead_time + review_period)` window before the new safety stock
figures are trusted.

**No retraining required.** The production model predicts one day ahead from real historical
features, not its own prior forecasts — periodic-review only changes the aggregation window
(2 weeks of daily forecasts vs. 1), not what the model needs to learn. 06d Section 5 already
tested training directly on a wider horizon (7-day target, Experiments A/C) and found it
regressed vs. summing daily forecasts — independent evidence this generalizes.

**Phase F (07/Fold 3) is blocked until Section 9 is complete and locked.**

---

## Phase F — Fold 3 Final Production Simulation
### Notebook: `07_fold3_final_evaluation.ipynb`

**This notebook is run exactly once. It is never rerun. The moment Fold 3
results are visible, the pipeline is locked regardless of what they show.**

Fold 3 is not a model tuning step. It is a production simulation: does the
full pipeline — routing, forecasting, uncertainty quantification, and
inventory simulation — behave consistently on unseen data?

**Section 1 — Pre-Flight Checklist**

All of the following must be confirmed before any cells execute:

```python
# ── Locked decisions from pre-Fold 3 work ─────────────────────────────────
# 06d tested direct 7-day target and asymmetric loss; both failed the adoption gate
# (regressed WAPE/p90 WAPE vs. baseline once a target-construction bug was fixed).
# Neither was adopted -- PRODUCTION_MODEL is the unmodified 06b baseline, saved under
# 06d's output filename (tweedie_optimized_fold2.txt) but architecturally identical
# to 06b. No OPTIMIZATION_METHOD to select.
PRODUCTION_MODEL    = 'tweedie_baseline'  # 06b model, unchanged -- see 06d Section 5
REGIME_THRESHOLDS   = {'adi': 1.32, 'cv2': 0.49}  # locked in 06c
CONFORMAL_ALPHA     = [value locked in 06e]
HYSTERESIS_WEEKS    = 4

# Pre-flight assertions
assert fold3_val['date'].min() == pd.Timestamp('2015-02-01'), 'Fold 3 start mismatch'
assert fold3_train['date'].max() < pd.Timestamp('2015-02-01'), 'Leakage detected'
print('Pre-flight assertions passed.')
```

**Section 2 — Demand Regime Classification on Fold 3 Training Data**

Recompute ADI and CV² from Fold 3 training window (Feb 2011 → Jan 2015).
The method is identical to 06c — the values are recomputed from the longer
training window, not copied from 06c.

Compare regime distribution: Fold 2 vs Fold 3. Flag SKUs that reclassified.
Apply hysteresis gate. Document stability.

**Section 3 — Retrain Production Model on Fold 3 Training Data**

Retrain the Tweedie model on the full Fold 3 training window using the frozen
06b/Experiment-0 hyperparameters -- 06d adopted none of its three experiments
(direct 7-day target, asymmetric loss, combined), so this is the original baseline
architecture, not a modified one. No Optuna, no adjustments.

**Section 4 — Conformal Calibration on Fold 3**

Refit conformal residuals from Fold 3 training in-sample predictions.
The method is identical to 06e. The residual distribution is recomputed —
not copied from 06e pkl files.

**Section 5 — Run Full Pipeline**

Apply complete pipeline to Fold 3 val:
1. Route each SKU by Fold 3 regime classification
2. Apply optimized Tweedie (Smooth/Erratic) or Croston/TSB (Intermittent/Lumpy)
3. Apply conformal intervals to Tweedie predictions
4. Run inventory simulation for **all three policies** — naive, static continuous-review
   (s,Q), and dynamic periodic-review — exactly as calibrated in 06g Sections 7 and 9.
   Reuse those formulas and parameters directly (safety stock, review period, lead time,
   `52/N_WEEKS_VAL` annualization convention); this is Fold 3 confirming a locked
   methodology, not re-deriving one. Save per-SKU `units_short`, `avg_inventory`, and
   `unit_cost` for all three policies, plus fill rate at q50/q75/q80/q90/q95/q99 for the
   app's service-level selector (Decision 2) — this is the one and only Fold 3 simulation
   run this notebook performs.

**Section 5b — Cost Views (reuses Section 5's output only, no new simulation)**

Everything below is arithmetic on the single simulation run above — no additional model
calls or Fold 3 touches, so this doesn't compromise "run exactly once." Apply 06g
Sections 8–9's exact cost formulas to Fold 3's per-SKU output:
- Dynamic vs. static-locked cost, 24-scenario real-cost grid (mirrors 06g Section 8)
- Dynamic vs. static's own cost-minimizing level, same 24-scenario grid (mirrors 06g
  Section 9) — this is the number that becomes the app's headline (see Section 7 below)

**Section 6 — Evaluate Against Fold 2 Benchmarks**

```
Fold 3 vs Fold 2 Stability Check:

Metric                          Fold 2      Fold 3      Stable?
Regime distribution (% Smooth)  [F2]%       [F3]%       ±5% = stable
Median per-SKU weekly WAPE      [F2]%       [F3]%       ±5% = stable
p90 per-SKU weekly WAPE         [F2]%       [F3]%       ±10% = stable
q90 empirical coverage          [F2]%       [F3]%       ±3pp = stable
Simulated stockout rate (q80)   [F2]%       [F3]%       ±5pp = stable
Simulated fill rate (q80)       [F2]%       [F3]%       ±5pp = stable
Dyn vs static-optimal (2.0x/25%)[F2]%       [F3]%       ±5pp = stable
```

The last row compares against 06g Section 9's locked default-scenario number, not a new
one — it's the cost-based counterpart to the fill-rate rows above and should move
together with them; a stable fill rate but an unstable cost number (or vice versa) is
itself worth flagging rather than averaging away.

**Section 7 — Final Headline Numbers**

**Primary headline — matches 06g Section 9's locked methodology, not the older single-
number framing this section originally specified:**

- Dynamic (periodic-review) vs. static's own cost-minimizing level, across the 24
  real-cost scenarios: X%–Y%, with static's chosen level per scenario reported
  ("best point in the tested q50–q99 range," never "confirmed cost optimum" — same
  caveat locked in 06g Section 4/9)
- Dynamic vs. naive, same 24-scenario grid: A%–B%
- One representative scenario stated in dollar terms for the README/app (e.g. the 2.0x/
  25% default): "$Z estimated annual savings vs. naive reorder"

**Supporting detail (still reported, no longer the headline):**
- Fill rate at q80 and q90, dynamic vs. naive, at equivalent average inventory
- % of SKUs forecast within 30% weekly error (forecastable SKUs)
- % improvement over SARIMA/Prophet on representative series

**Section 8 — Fold 3 Signoff**

One paragraph. State the numbers. State that the pipeline is locked.
This notebook is never run again.

**Outputs:**
- `tweedie_optimized_fold3.txt`
- `sku_regimes_fold3.parquet`
- `conformal_residuals_fold3.pkl`
- `final_predictions_fold3.parquet`
- `simulation_results_fold3.parquet`
- `dynamic_cost_sensitivity_fold3.parquet` (Section 5b, mirrors 06g Section 8)
- `static_vs_dynamic_headtohead_fold3.parquet` (Section 5b, mirrors 06g Section 9 —
  this is the app's headline source)

---

## Phase G — Explainability
### Notebook: `08_explainability.ipynb`

Uses the optimized Tweedie model from Fold 3.

**Section 1 — SHAP Global Importance**

SHAP on the final production Tweedie model. Does feature ranking make
business sense? Compare against 05b v1 feature importance — document
which features improved in rank due to v2 engineering.

**Section 2 — SHAP by Demand Regime**

This is the key insight for the portfolio: different SKU types are
driven by different features. Run SHAP separately per regime:

- Smooth SKUs: likely dominated by price and recent lag features
- Erratic SKUs: likely dominated by promotional/calendar features
- Intermittent SKUs: likely dominated by seasonality and event proximity

Showing this explicitly demonstrates you understand the demand drivers
change by SKU type — not just a global importance bar chart.

**Section 3 — Feature Stability Across Departments**

Are the top 5 SHAP features consistent across FOODS, HOBBIES, HOUSEHOLD?
If a feature ranks #1 globally but #8 in HOBBIES, that is worth noting.

**Section 4 — Per-SKU Waterfall Plots**

Three representative SKUs — one per forecastable regime:
- FOODS_3_163_CA_3 (Smooth — benchmark series)
- One Erratic SKU (high-volume, volatile demand)
- One Intermittent SKU (for contrast — show Croston output vs Tweedie)

**Section 5 — Anomaly Detection on Fold 3 Residuals**

- Z-score flags: demand spikes (z > 3), suppressed demand (z < −3)
- Flag SKUs with persistent residual bias (systematic over or under
  in a specific month) — these are candidates for regime reclassification
  in a production monitoring system

---

## Phase H — App Data Preparation
### Notebook: `09_app_data_prep.ipynb`

Pre-compute everything the app needs. The app does zero model inference
at runtime — all forecasts and simulations are precomputed and stored as
parquet.

**Outputs to precompute:**
- Fold 3 predictions for all SKUs at all service levels (q50/q75/q80/q90/q95/q99)
- Reorder parameters per SKU at 7-day and 14-day lead times
- Simulation results per SKU (stockout rate, fill rate, avg inventory)
- SHAP values per SKU (top 5 features + values for waterfall)
- Regime classification per SKU
- SKU metadata (department, store, category, mean demand, zero rate)

Load time target: app loads in < 3 seconds. All heavy computation happens here.

---

## Deployment Layer — Streamlit App (`app.py`)

Five pages. Any SKU selectable on any page. All data precomputed.

---

### Page 1 — SKU Inventory Dashboard

The primary page. Answers: "What should I do about this SKU right now?"

**Inputs (sidebar):**
- Store / SKU selector (any individual `item_id × store_id`)
- Current inventory level (units)
- Supplier lead time (7 days default, configurable)
- Target service level (80% / 90% / 95% / 99%)

**Outputs:**
- 28-day demand forecast chart with selected service level band
- Regime badge (Smooth / Erratic / Intermittent / Lumpy)
- Forecast confidence score (derived from interval width ÷ mean demand —
  high = narrow intervals = confident, low = wide intervals = uncertain)
- Reorder recommendation: quantity and timing
- Projected stockout date at current inventory level
- Safety stock suggestion for selected service level
- If Intermittent/Lumpy: "This SKU uses [Croston/policy]-based forecasting.
  The model-based approach is not reliable for this demand pattern."

**Aggregation support:**
Filter to store, department, category, or state level for planning views.
Point forecasts aggregate by sum. Quantile bands at aggregate levels are
labeled "indicative" — summing q90 across SKUs overstates aggregate q90
due to demand diversification.

---

### Page 2 — Portfolio Risk Monitor

Answers: "Which SKUs need attention this week?"

- Full SKU table: forecast, demand ratio, regime, WAPE tier, stockout risk flag
- Filterable by store, department, regime, risk level
- Red highlight: SKUs projected to stockout within lead time at current
  inventory
- Yellow highlight: SKUs with recent forecast error spike (rolling WAPE
  increased > 20% last 4 weeks)
- Regime change alerts: SKUs that reclassified in the most recent window

---

### Page 3 — Backtesting and Simulation Results

Answers: "Why should I trust this system?"

- Walk-forward CV results table (all models: Naive, SARIMA, Prophet,
  XGBoost v1, LightGBM Tweedie)
- Fold 3 inventory simulation results: fill rate and stockout rate at each
  service level, vs naive reorder baseline
- Cost tradeoff curve: total inventory cost vs service level
- Coverage validation table: does q90 actually cover 90%?

---

### Page 4 — Explainability Engine

Answers: "Why is the model forecasting this?"

- Global SHAP feature importance bar chart
- Per-SKU SHAP waterfall (linked to SKU selector — same SKU as Page 1)
- Top 3 demand drivers in plain English for selected SKU:
  "Demand for this SKU is primarily driven by: recent sales trend (+),
  day of week (Fridays peak), proximity to SNAP payment dates (+)"
- Price elasticity visualization for selected product-store combination

---

### Page 5 — Technical Deep Dive

For the technical interviewer.

- Full metric tables: per-fold log-RMSE, WAPE, bias by model
- Quantile calibration summary: empirical vs target coverage per level
- Per-SKU WAPE distribution histograms by regime
- Error distributions by department and store
- SHAP feature importance by regime (from 08)
- Methodology note on conformal prediction and Syntetos-Boylan classification

---

## Production System Design (README Section)

*Demonstrates understanding of real-world ML systems.*

### Retraining Pipeline
- Weekly recompute: ADI + CV² per SKU from expanding training window
- Regime reclassification with 4-week hysteresis gate
- Monthly model retrain on expanding window (frozen hyperparams)
- Conformal residuals refit from most recent N weeks of production data

### Monitoring
- Rolling per-SKU WAPE tracked weekly — alert if p90 drifts > 10%
- Quantile coverage monitoring — alert if q90 drops below 85%
- Regime change detection — flag SKUs transitioning regimes
- Demand spike detection via residual z-scores

### Inference
- Batch forecasting at 7, 28, 90-day horizons
- All predictions stored as parquet for downstream consumption
- Zero model inference at query time

---

## Execution Order and Time Estimates

| Step | Notebook | Est. time | Blocking dependency |
|---|---|---|---|
| 0 | Lock `06b` winner decision | 30 min | None — do immediately |
| 1 | `06c` SKU audit + regime classification | 2–3 hrs | 06b locked |
| 2 | `06d` Model optimization | 3–4 hrs | 06c complete |
| 3 | `06e` Conformal prediction | 2–3 hrs | 06d complete |
| 4 | `06f` Intermittent demand (Croston/TSB) | 2–3 hrs | 06c complete (parallel with 06d/06e) |
| 5 | `06g` Inventory simulation | 3–4 hrs | 06e + 06f complete |
| 6 | `07` Fold 3 — run once | 3–4 hrs | ALL above locked |
| 7 | `08` Explainability | 2–3 hrs | 07 complete |
| 8 | `09` App data prep | 1–2 hrs | 07 + 08 complete |
| 9 | `app.py` Streamlit | 2–3 days | 09 complete |
| 10 | README + portfolio | Half day | All above complete |

**Total modeling work before app:** ~5–6 focused sessions.

`06f` (Croston/TSB) can run in parallel with `06d` and `06e` since it only
depends on `06c` regime assignments and does not require the optimized
Tweedie model.

---

## Key Decisions Locked — Do Not Revisit

| Decision | Rationale |
|---|---|
| Production model: LightGBM Tweedie | Best per-SKU demand ratio, unit-space predictions, no retransformation bias |
| Global calibration: rejected | Overcorrects at series level. Demand ratio 2.3–2.5× median post-calibration |
| Department corrections: rejected | Same architectural flaw as global calibration at finer granularity |
| SKU routing: Syntetos-Boylan ADI/CV² | Data property — no ground truth needed, recomputes on expanding window, industry standard |
| Uncertainty: conformal prediction | Coverage-guaranteed intervals from one model, no quantile crossing, all service levels |
| Intermittent SKUs: Croston/TSB | Architecturally correct for zero-inflated intermittent demand. ML not the right tool here |
| Optimization: evaluate on per-SKU weekly WAPE | Aggregate metrics mask SKU-level behavior and are useless for inventory decisions |
| Fold 3: run once, results locked | Non-negotiable. Any retrain after seeing Fold 3 is leakage |
| Hysteresis: 4-week gate | Prevents regime boundary flip-flopping in production |

---

## The Portfolio Narrative at the End

The repository tells a complete, auditable story:

1. **EDA** — understand demand structure and sparsity
2. **Baselines** — SARIMA and Prophet establish signal ceiling
3. **Feature engineering v1 → v2** — principled improvement from diagnostics
4. **XGBoost v1 → v2 → LightGBM Tweedie** — model progression with documented rationale
5. **Model selection** — Tweedie wins; global calibration rejected by evidence
6. **SKU audit** — demand regime classification, not aggregate metrics
7. **Optimization** — targeted improvement on the right SKU segment
8. **Uncertainty quantification** — conformal prediction, coverage guaranteed
9. **Intermittent handling** — Croston/TSB where ML cannot forecast reliably
10. **Inventory simulation** — stockout rate and fill rate on holdout data
11. **Fold 3** — one clean production simulation, never touched before this moment
12. **Explainability** — SHAP confirms model learns the right demand drivers
13. **App** — any SKU, any service level, reorder recommendation in < 60 seconds

**The headline you say in every interview:**

> "On the Fold 3 holdout, the dynamic periodic-review policy beats a traditional
> static reorder policy even at static's own best-case service level, by [X]%–[Y]%
> across a real-cost sensitivity grid — not just at naive or an arbitrarily-chosen
> service level. Estimated annual savings of $Z at representative retail cost
> assumptions. Intermittent SKUs are handled via Croston/TSB rather than forcing
> a global ML model onto data with insufficient signal."

Fill in X, Y, Z from Phase E (Fold 2) simulation results as soon as 06g is
complete. Do not wait for Fold 3. Fold 2 numbers anchor the story; Fold 3
confirms it.

## Minimum Viable Production System

*Answer this question before every interview: "If you had two weeks, what
ships?"*

The irreducible core of this system is four components. Everything else is
supporting analysis and diagnostics:

1. **LightGBM Tweedie** — point forecasts for Smooth and Erratic SKUs
2. **Croston/TSB routing** — principled forecasts for Intermittent/Lumpy SKUs
   where the global model has no signal
3. **Conformal prediction wrapper** — coverage-guaranteed intervals from one
   calibration step, no distributional assumptions
4. **Inventory simulation layer** — converts forecasts to reorder decisions
   and measures business outcomes, not just accuracy

Every other notebook (optimization branches, SHAP by regime, coverage
validation tables, anomaly detection) strengthens the case for these four
components. None of them are the system. If an interviewer asks "what would
you cut?", the answer is: the optimization search in 06d becomes a fixed
hyperparameter set, and the SHAP analysis becomes a single global importance
chart. The four components above do not get cut.

**The whiteboard version (practice this):**

Raw sales data
↓
ADI + CV² per SKU  →  Regime classification (Syntetos-Boylan)
↓                        ↓
Smooth/Erratic           Intermittent/Lumpy
↓                        ↓
LightGBM Tweedie          Croston/TSB
↓                        ↓
Conformal intervals      Native uncertainty
↓                        ↓
└──────────┬─────────────┘
↓
Reorder point + safety stock
↓
Inventory simulation → Fill rate, stockout rate, cost

---

Every notebook is a frozen artifact. Every decision is documented and justified.
The before-and-after narrative is built into the repository structure.
The system answers the question that matters in production:
**not how accurate is the forecast, but how good are the inventory decisions.**