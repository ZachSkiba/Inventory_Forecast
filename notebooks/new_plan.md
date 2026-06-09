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
SKUs. Do not attempt to fix Intermittent or Lumpy SKUs here — they are
handled in Phase D.

**All optimization is evaluated on per-SKU weekly WAPE distribution, not
aggregate metrics. An optimization is only adopted if it improves the
median per-SKU weekly WAPE without worsening the p90.**

**Section 1 — Baseline: Tweedie Suppressed on Smooth + Erratic SKUs Only**

Re-evaluate the 06b Tweedie suppressed predictions filtered to Smooth +
Erratic SKUs only. This is the correct baseline — prior evaluations mixed
in Intermittent and Lumpy SKUs which dragged aggregate metrics down.

Document: median weekly WAPE, p75, p90, % SKUs < 30%, % SKUs < 50%,
% SKUs > 100% — on Smooth + Erratic SKUs only.

**Section 2 — Optimization Option A: Direct Multi-Step Target**

**What it is:** Retrain Tweedie with the 7-day forward sum as the target
instead of next-day demand. This eliminates error accumulation from
recursive daily forecasting and directly optimizes the signal used for
weekly inventory decisions.

**Implementation:**
- Modify feature engineering to produce `target_7d = sum of next 7 days demand`
- Retrain Tweedie on Fold 2 training data with new target
- Evaluate per-SKU weekly WAPE on Smooth + Erratic SKUs

**Expected impact:** Should improve weekly WAPE directly since the model
is now trained on the exact aggregation level used for reorder decisions.

**Section 3 — Optimization Option B: Asymmetric Loss Function**

**What it is:** Replace the symmetric Tweedie loss with a custom objective
that penalizes underprediction more heavily than overprediction. Directly
encodes the business reality that stockouts cost more than carrying excess
inventory.

**Implementation:**

```python
def asymmetric_loss(y_true, y_pred, alpha=0.6):
    """
    alpha > 0.5 penalizes underprediction more than overprediction.
    alpha = 0.6 means underprediction errors are weighted 1.5× overstock errors.
    Tune alpha on Fold 2 per-SKU demand ratio distribution.
    """
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2 * alpha * residual,
                                  -2 * (1 - alpha) * residual)
    hess = np.where(residual > 0, 2 * alpha,
                                  2 * (1 - alpha))
    return grad, hess
```

Tune `alpha` on Fold 2 val. Target: shift median per-SKU demand ratio
toward 0.90–0.95 without pushing p90 above 1.5.

**Section 4 — Comparison and Selection**

Run both options. Compare against baseline on:
- Median per-SKU weekly WAPE
- p90 per-SKU weekly WAPE (tail behavior)
- Median per-SKU demand ratio
- % SKUs with demand ratio in 0.8–1.2 band

Select the option that best improves the distribution without degrading the
tail. If neither clears a 5% improvement in median weekly WAPE, retain the
06b Tweedie suppressed model as-is and document the result. Do not force an
improvement that is not there.

**Outputs:**
- `tweedie_optimized_fold2.txt` — winning optimized model
- `tweedie_optimization_results.pkl` — comparison table for portfolio

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

## Phase D — Intermittent Demand Handling
### Notebook: `06f_intermittent_demand.ipynb`

**Purpose:** Provide principled forecasts for Intermittent and Lumpy SKUs
that the global Tweedie model cannot handle. Do not apply ML where the
signal does not support it.

**Section 1 — Croston's Method for Intermittent SKUs**

Croston's method separates demand into two components:
- **Demand size:** exponential smoothing on the magnitude of demand when it
  occurs (nonzero observations only)
- **Demand interval:** exponential smoothing on the number of periods between
  demand events

Point forecast = smoothed demand size ÷ smoothed demand interval.
This gives "expected demand per period" accounting for the intermittent pattern.

**TSB variant (Terence-Syntetos-Boylan):** Improves on original Croston by
letting demand probability decay when no demand is observed. Better for SKUs
that may be dying or seasonal. Use TSB as default.

```python
def tsb_forecast(demand: np.ndarray, alpha: float = 0.1,
                 beta: float = 0.1) -> float:
    """
    TSB forecast for intermittent demand.
    alpha: smoothing for demand probability
    beta:  smoothing for demand size
    Returns: expected demand per period
    """
    p = (demand > 0).mean()   # initialize demand probability
    z = demand[demand > 0].mean() if (demand > 0).any() else 0.0
    for d in demand:
        if d > 0:
            p = (1 - alpha) * p + alpha * 1.0
            z = (1 - beta)  * z + beta  * d
        else:
            p = (1 - alpha) * p
    return p * z
```

Tune alpha and beta on Fold 2 training data per SKU via grid search.
Evaluate on Fold 2 val: per-SKU weekly WAPE on Intermittent SKUs.

**Section 2 — Historical Policy for Lumpy SKUs**

Truly unforecastable SKUs (Lumpy regime: high ADI, high CV²) should not
receive an ML or statistical forecast. The correct approach is a min-max
inventory policy driven by historical demand.

```python
def historical_policy_forecast(train_demand: pd.Series,
                                lead_time_weeks: int = 1,
                                buffer_multiplier: float = 1.25) -> dict:
    """
    Min-max reorder policy for Lumpy SKUs.
    Reorder point: max weekly demand in same calendar period last year × buffer
    Order quantity: bring inventory up to max observed weekly demand × buffer
    """
    weekly_demand = train_demand.resample('W').sum()
    max_weekly    = weekly_demand.max()
    avg_weekly    = weekly_demand.mean()
    reorder_point = max_weekly * lead_time_weeks * buffer_multiplier
    order_qty     = max_weekly * buffer_multiplier - avg_weekly
    return {
        'reorder_point': reorder_point,
        'order_qty':     max(order_qty, avg_weekly),
        'max_weekly':    max_weekly,
        'avg_weekly':    avg_weekly,
    }
```

**Section 3 — Evaluation: Croston vs Tweedie on Intermittent SKUs**

Direct comparison on the Fold 2 val set for Intermittent SKUs only:
- Per-SKU weekly WAPE: Croston/TSB vs Tweedie suppressed
- Demand ratio: Croston/TSB vs Tweedie suppressed
- Show that Croston/TSB improves over the global model on this segment

This is a key portfolio moment — demonstrating that you know when not to
use ML and apply a domain-appropriate method instead.

**Section 4 — Unified Prediction Output**

Combine all routing paths into a single predictions dataframe:

```python
# final_predictions_fold2.parquet schema:
# id | date | regime | routing | point_forecast |
# q50 | q75 | q80 | q90 | q95 | q99 |
# interval_source ('conformal' / 'croston_native' / 'policy')
```

For Croston SKUs: q50 = TSB point forecast; q90 = point forecast +
2× std of nonzero demand observations.
For Lumpy SKUs: all quantiles = historical policy reorder point.

**Outputs:**
- `croston_predictions_fold2.parquet`
- `final_predictions_fold2.parquet` — unified, all SKUs, all service levels

---

## Phase E — Inventory Simulation
### Notebook: `06g_inventory_simulation.ipynb`

**Purpose:** Convert forecasts → reorder decisions → simulate business
outcomes on the Fold 2 val window. Produce the headline business metrics
that anchor the portfolio narrative.

**Section 1 — Reorder Parameter Computation**

For each SKU, compute reorder parameters from the unified predictions:

```python
def compute_reorder_params(sku_id: str,
                            point_forecast_weekly: float,
                            q90_upper: float,
                            lead_time_weeks: int,
                            service_level: float) -> dict:
    """
    Compute reorder point and safety stock for a single SKU.

    safety_stock  = (q_upper - point_forecast) × sqrt(lead_time)
                  = interval half-width scaled by lead time uncertainty
    reorder_point = point_forecast × lead_time + safety_stock
    reorder_qty   = reorder_point - current_inventory (when triggered)
    """
    interval_width = q90_upper - point_forecast_weekly
    safety_stock   = interval_width * np.sqrt(lead_time_weeks)
    reorder_point  = point_forecast_weekly * lead_time_weeks + safety_stock
    return {
        'sku_id':         sku_id,
        'point_forecast': point_forecast_weekly,
        'safety_stock':   safety_stock,
        'reorder_point':  reorder_point,
        'service_level':  service_level,
    }
```

**Section 2 — Inventory Depletion Simulation**

Simulate week-by-week inventory depletion on the Fold 2 val window
(Feb 2014 → Jan 2015) using actual sales as ground truth.

```python
def simulate_inventory(actual_weekly_demand: np.ndarray,
                        reorder_point: float,
                        reorder_qty: float,
                        initial_inventory: float,
                        lead_time_weeks: int) -> dict:
    """
    Simulate inventory trajectory under the reorder policy.
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

Run the simulation at q50, q75, q80, q90, q95 for all SKUs.
Aggregate results by regime and service level:

```
Simulation Results — Fold 2 Val (52 weeks, all SKUs):

Service level   Stockout rate   Avg inventory   Fill rate
q50             [result]%       [result] units  [result]%
q75             [result]%       [result] units  [result]%
q80             [result]%       [result] units  [result]%
q90             [result]%       [result] units  [result]%
q95             [result]%       [result] units  [result]%

Naive reorder   [result]%       [result] units  [result]%
(reorder on mean historical demand — no forecast)
```

**This table is the headline result of the entire project.**

**Section 4 — Cost Tradeoff Curve**

Plot: as service level increases from q50 to q99, overstock cost increases
and stockout cost decreases. Find and annotate the crossover point — the
service level at which total inventory cost is minimized.

Cost parameters (locked for reporting — configurable in app):
- Median unit cost: $4.00 (M5 dataset approximate)
- Stockout cost: 2× unit cost per unit short ($8.00) — represents lost sale
  plus estimated customer churn penalty
- Carrying cost: 25% of unit cost per year ($1.00/unit/year) — standard
  retail holding cost assumption

From these, compute and report in the notebook summary cell:

  total_stockout_cost   = stockout_units × $8.00
  total_carrying_cost   = avg_inventory_units × $1.00
  total_inventory_cost  = total_stockout_cost + total_carrying_cost

Report the dollar figures at q80 vs naive baseline. This is the headline
business number for the README and every interview: "At q80 service level,
the system reduces total inventory cost by approximately $X vs naive reorder
on the Fold 2 validation window."

**Section 5 — Simulation by SKU Regime**

Break out simulation results by regime (Smooth, Erratic, Intermittent,
Lumpy). Confirm:
- Smooth SKUs achieve high fill rates at low service levels (model is accurate)
- Lumpy SKUs require high service levels or policy-based buffers to achieve
  acceptable fill rates

**Outputs:**
- `reorder_parameters_fold2.parquet`
- `simulation_results_fold2.parquet`
- Headline numbers locked for README and app

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
PRODUCTION_MODEL    = 'tweedie_optimized'  # from 06d
OPTIMIZATION_METHOD = '[direct_multistep or asymmetric_loss]'  # from 06d
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

Retrain the optimized Tweedie model on the full Fold 3 training window
using frozen params from 06d. No Optuna, no adjustments.

**Section 4 — Conformal Calibration on Fold 3**

Refit conformal residuals from Fold 3 training in-sample predictions.
The method is identical to 06e. The residual distribution is recomputed —
not copied from 06e pkl files.

**Section 5 — Run Full Pipeline**

Apply complete pipeline to Fold 3 val:
1. Route each SKU by Fold 3 regime classification
2. Apply optimized Tweedie (Smooth/Erratic) or Croston/TSB (Intermittent/Lumpy)
3. Apply conformal intervals to Tweedie predictions
4. Run inventory simulation at q80 and q90

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
```

**Section 7 — Final Headline Numbers**

State the production system's performance in business terms:

- At q80 service level and 7-day lead time, simulated fill rate on
  Fold 3 holdout: X%
- Vs naive reorder baseline: Y% fill rate at equivalent inventory level
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

> "At 90% service level and 7-day lead time, the system achieves [X]% fill rate
> on the Fold 3 holdout — [Y] percentage points better than naive reorder at
> equivalent average inventory, reducing estimated total inventory cost by $Z
> on the validation window. Intermittent SKUs are handled via Croston/TSB
> rather than forcing a global ML model onto data with insufficient signal."

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