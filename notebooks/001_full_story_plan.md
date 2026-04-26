# Full Story Notebook — Retail Demand Forecasting (M5 Walmart)
# Presentation Plan for 30-Minute Session

---

## How to use this plan

Each section below becomes one or more cells in the full story notebook.
Sections marked **[CHART]** require a saved figure — all charts go in:

```
outputs/presentation_charts/
```

Create this folder before starting. Every chart gets a filename listed
next to the tag so you know exactly what to save and where.

Estimated speaking time is noted per section. Total target: 32–35 minutes
with natural pauses. The notebook scrolls top to bottom — no jumping around.

---

---

# PART 1 — THE PROBLEM
*Estimated time: 4 minutes*

---

## Section 1.1 — What does Walmart actually need?

**What to say:** Start with the business reality before any math.
Walmart runs 10,000+ stores. Every store has thousands of products.
Every product needs to be on the shelf at the right time. If you
over-order, product expires or takes up shelf space. If you under-order,
the shelf is empty and you lose the sale entirely — and possibly the
customer. This is called a stockout.

**Key definitions to include:**
- **Demand forecasting:** predicting how many units of a product a
  specific store will sell in a future time period
- **Stockout:** a product is out of stock when a customer wants to buy it
- **Overstock:** too much inventory ordered, leading to waste or markdowns
- **The business question:** given everything we know about a product's
  history, its price, upcoming holidays, and local events — how many
  units will store X sell next week?

**Why this is hard:** You are not forecasting one product. You are
forecasting 30,490 product-store combinations simultaneously, each with
its own demand pattern, price history, and local context.

---

## Section 1.2 — The dataset: M5 Walmart

**What to say:** Introduce the M5 dataset — a real Walmart sales dataset
used in a Kaggle competition in 2020. This is not simulated data.

**Key facts to state:**
- 30,490 unique product-store series
- 3,049 products × 10 stores across 3 states (CA, TX, WI)
- 5.25 years of daily sales: Jan 29 2011 → Apr 24 2016
- Three source files: sales (units), calendar (dates/events/SNAP),
  prices (weekly sell price per product per store)
- Revenue is derived: `units_sold × sell_price` — not directly observed

**[CHART 1]** `01_dataset_overview.png`
A simple table or visual showing the three files, what each contains,
and how they join together. Like a schema diagram.

**Key definition:**
- **Wide format vs long format:** the sales file has one row per product
  and one column per day (1,913 columns). We reshape it to long format —
  one row per product-store-day — before any analysis is possible.
  This produces 58 million rows.

---

## Section 1.3 — Why this dataset is hard: zero-inflation

**What to say:** Before building any model, we need to understand the
shape of the data. The first thing you notice is that most product-store
days have zero sales.

**Key number:** 68.2% of all product-store-days have zero units sold.

**What this means:** If you pick any specific product at any specific
store on any given day, there is a 68% chance it sold nothing. This is
not a data quality issue — it is the nature of retail demand at the
individual series level.

**Two types of zeros — this distinction matters:**
- **Intermittent demand zeros:** the product exists but just didn't sell
  that day (normal sparse demand)
- **Structural zeros:** the product wasn't stocked at that store yet,
  or was discontinued. Mean longest zero streak per series: 430 days.
  44% of series have a streak longer than 365 consecutive days.

**Why this matters for modeling:** You cannot use lag features that
reach back across a structural zero gap — that would be lagging into a
period when the product didn't exist, which produces meaningless signal.

**[CHART 2]** `02_zero_inflation.png`
Histogram of units sold (non-zero days only) showing the heavy
right skew. Include the 68.2% stat as an annotation.

---

---

# PART 2 — EXPLORATORY DATA ANALYSIS
*Estimated time: 7 minutes*

---

## Section 2.1 — EDA philosophy: every finding drives a decision

**What to say:** EDA is not just exploring for curiosity. Every finding
in our EDA directly determined a modeling choice. Walk through the four
findings that mattered most.

---

## Section 2.2 — Finding 1: Revenue structure

**What to say:** Before modeling anything we need to understand where
the money is. Not all products and stores are equal.

**Key findings:**
- FOODS dominates at 58% of total revenue ($108.9M)
- FOODS_3 alone is 37.8% of all revenue — single largest department
- CA_3 is the highest-revenue store at 17.1% — nearly 3× the smallest
  store (CA_4 at 6.5%)
- HOBBIES_2 is negligible at 0.63%

**Modeling decision this drives:** We model FOODS series as our
representative individual series. HOBBIES_2 is too sparse. Store encoding
is essential — CA_3 and CA_4 are in the same state but behave completely
differently.

**[CHART 3]** `03_revenue_by_category.png`
Side-by-side horizontal bar charts: revenue by category and revenue
by department. Already generated in EDA Section 7.

---

## Section 2.3 — Finding 2: Trend and seasonality

**What to say:** The aggregate monthly revenue series is the foundation
for SARIMA and Prophet. We need to understand its structure before
choosing any model parameters.

**Key findings:**
- Strong upward trend: ~$2M/month in 2011 → ~$4M/month by 2016
- Seasonality is subtle — August (+$147K) and March (+$128K) are the
  strongest months, December slightly negative (stores close Christmas)
- Trend dominates over seasonality — this matters for model choice

**Definition — seasonal decomposition:** We can mathematically separate
a time series into three components:
- **Trend:** the underlying direction (up, down, flat)
- **Seasonality:** the repeating calendar pattern (same every 12 months)
- **Residual:** everything left over that the model cannot explain

**[CHART 4]** `04_seasonal_decomposition.png`
The 4-panel seasonal decomposition plot from EDA Section 10.
Observed / Trend / Seasonal / Residual.

**Modeling decision this drives:** We need a model that handles both
trend and seasonality. The trend is strong and persistent — a model
that mean-reverts will underperform at longer horizons.

---

## Section 2.4 — Finding 3: SNAP and holiday effects

**What to say:** The calendar contains two major external demand drivers
that a pure time series model cannot see.

**SNAP (Supplemental Nutrition Assistance Program):**
- US government food assistance benefits distributed on specific days
- Different schedule per state — CA, TX, and WI have different SNAP days
- Effect on FOODS revenue:
  - WI FOODS: +32.5% on SNAP days vs non-SNAP days
  - TX FOODS: +17.2%
  - CA FOODS: +10.3%

**Holiday effects (key findings from EDA):**
- Labor Day: +19.6% vs baseline
- Super Bowl: +18.9% vs baseline
- Christmas: -100% (stores closed)
- Thanksgiving: -40.8% (traffic shifts to day before)

**Key insight:** Christmas and Thanksgiving signal lives in the days
*before* the holiday, not on the day itself. We need lead features,
not just a binary flag on the holiday date.

**[CHART 5]** `05_snap_effect.png`
Bar chart showing SNAP vs non-SNAP revenue uplift per state,
separated for FOODS category. Already computed in EDA Section 12.

**Modeling decision this drives:** SNAP flags (`snap_CA`, `snap_TX`,
`snap_WI`) must be state-level features in XGBoost — a single global
flag loses the different magnitudes per state. This is one of the
strongest features in the dataset.

---

## Section 2.5 — Finding 4: Price elasticity

**What to say:** Prices change week over week in this dataset.
Understanding how price changes affect demand tells us which price
features to engineer.

**Key finding — asymmetric price elasticity:**
At the individual product-store level (FOODS_3_383 @ CA_3):
- Price drops: r = 0.553 — a 10% price drop is associated with ~77%
  demand increase
- Price increases: r = 0.180 — much weaker response

**Definition — price elasticity:** how sensitive demand is to a change
in price. Asymmetric means customers respond differently to price drops
vs price increases — they stock up aggressively on discounts but don't
proportionally cut back when prices rise.

**Important caveat:** At the department-aggregate level, correlations
are near zero (r < 0.13). This is confounding — Walmart discounts
slow-moving products, so price drops and low demand co-occur. The true
elasticity only appears at the product-store level.

**[CHART 6]** `06_price_elasticity.png`
The 6-panel elasticity scatter plots from EDA Section 18 showing
the three levels of granularity (all depts / FOODS_3 @ CA_3 /
single product). Already generated.

**Modeling decision this drives:** Price features must be engineered at
`item_id + store_id` level only. We need signed price change features
(`price_drop_pct` separate from `price_increase_pct`) because the
asymmetry is real.

---

## Section 2.6 — Finding 5: Series completeness

**What to say:** Before we can model individual product-store series,
we need to know how many series have enough history to be usable.

**Key finding:** Only 8.1% of the 30,490 series (2,469 series) have
sales in all 64 months. The other 91.9% have structural zero gaps —
products introduced, discontinued, or not stocked at specific stores.

**Modeling decision this drives:**
- SARIMA is only viable on the 2,469 complete series
- For the representative series we pick from FOODS_3 (highest revenue
  department) at CA_3 (highest revenue store) — the series with the
  most signal
- XGBoost handles incomplete series but lag features must not span gaps

**[CHART 7]** `07_series_completeness.png`
Histogram of active months per series from EDA Section 15.

---

---

# PART 3 — STATIONARITY AND MODEL SELECTION
*Estimated time: 5 minutes*

---

## Section 3.1 — What is stationarity and why does it matter?

**What to say:** Before we can fit SARIMA, we need to understand one
fundamental requirement.

**Definition — stationarity:** A time series is stationary if its
statistical properties (mean, variance) do not change over time. A
series with an upward trend is NOT stationary — the mean keeps rising.

**Why SARIMA requires it:** SARIMA's math assumes the series fluctuates
around a stable mean. If the mean keeps drifting upward, the model's
equations break down.

**How we fix it — differencing:**
- First differencing: subtract each value from the previous one.
  Converts "revenue levels" into "revenue changes"
- Seasonal differencing: subtract the value from the same month one
  year ago. Removes annual drift.

**Definition — ADF test (Augmented Dickey-Fuller):**
A statistical test for stationarity. p-value < 0.05 means stationary
(safe to model). p-value > 0.05 means non-stationary (needs differencing).

**Our results:**
| Series | Raw p-value | After seasonal diff |
|---|---|---|
| Aggregate | 0.318 (non-stationary) | 0.000 (stationary) |
| Representative | 0.0004 (already stationary) | — |

**Modeling decision:** Aggregate SARIMA uses D=1 (seasonal differencing).
Representative SARIMA uses D=0 (already stationary).

---

## Section 3.2 — How we determine SARIMA parameters: ACF and PACF

**What to say:** Once we know how much differencing to apply, we need
to determine the AR and MA orders. This is done with ACF and PACF plots.

**Definition — ACF (Autocorrelation Function):**
Measures how correlated the series is with its own past values at each
lag. A spike at lag 3 means "the value 3 months ago is correlated with
today's value."

**Definition — PACF (Partial Autocorrelation Function):**
Same as ACF but strips out the indirect effects of intermediate lags.
Isolates only the direct relationship at each distance.

**How to read them:**
- Significant PACF spikes at early lags → AR order (p)
- Significant ACF spikes at early lags → MA order (q)
- Spikes at lag 12 → seasonal AR (P) or seasonal MA (Q) terms needed

**Our readings:**
- Aggregate: PACF cuts off at lag 1 → p=1 candidate; no lag-12 spikes
- Representative: PACF lag-1 only, ACF cuts off at lag-2 → p=1, q=1

**Then AIC grid search confirms the final order.**

**Definition — AIC (Akaike Information Criterion):**
A model selection score that rewards goodness of fit while penalizing
unnecessary complexity. Lower AIC = better model. We grid search all
combinations within the candidate ranges and pick the lowest AIC.
We use AIC instead of a validation set because with only 48 training
months, holding out additional data would cost too many observations.

**[CHART 8]** `08_acf_pacf.png`
The 4-panel ACF/PACF plot from EDA Section 21. Both series,
after their respective differencing transformations.

---

## Section 3.3 — SARIMA: what it is and how it works

**What to say:** Now we have all the tools to understand SARIMA.

**Definition — SARIMA(p,d,q)(P,D,Q)[m]:**
Seasonal AutoRegressive Integrated Moving Average. Six parameters:

| Parameter | What it controls |
|---|---|
| p | How many past values predict today (AR order) |
| d | How many times to difference to remove trend |
| q | How many past errors correct today's forecast (MA order) |
| P | Seasonal AR — same as p but at the seasonal lag |
| D | Seasonal differencing — removes annual drift |
| Q | Seasonal MA — corrects for last year's same-month error |
| m | Length of one seasonal cycle (m=12 for monthly data) |

**How AR works:** "This month's revenue is a weighted sum of the last
p months plus noise." Captures momentum.

**How MA works:** "If I over-predicted last month by $50K, I correct
this month's forecast downward." Self-correcting.

**How differencing works:** Subtracts past values to remove trend,
so the model works with changes rather than levels.

**Our final models:**
- Aggregate: SARIMA(2,0,1)(0,1,1)[12]
- Representative: SARIMA(0,0,1)(0,1,1)[12]

---

## Section 3.4 — Prophet: what it is and how it differs from SARIMA

**What to say:** Prophet is a completely different approach to the
same problem. Instead of one equation, it fits three interpretable
components separately.

**The Prophet equation:**
```
y(t) = trend(t) + seasonality(t) + holidays(t) + noise
```

**Trend:** Models the underlying direction as a piecewise linear curve.
Prophet automatically detects changepoints — moments where the slope
shifts — and adjusts. No manual differencing required.

**Seasonality:** Models repeating calendar patterns using Fourier series
(sums of sine and cosine waves). For monthly data with annual seasonality,
Prophet fits yearly waves automatically.

**Holidays:** Accepts a dataframe of named events. Prophet fits a
separate effect for each event, directly incorporating the holiday
impacts we quantified in the EDA.

**Key hyperparameters we tuned:**

| Parameter | What it controls | Our value |
|---|---|---|
| `changepoint_prior_scale` | How flexible the trend is. Low = stiff straight line. High = wiggly. | 0.5 (aggregate), 0.01 (individual) |
| `seasonality_mode` | Additive = fixed dollar seasonal swing. Multiplicative = seasonal swing grows with revenue level. | multiplicative (aggregate), additive (individual) |
| `seasonality_prior_scale` | How strongly seasonal patterns are fitted | 1.0 |

**How we selected hyperparameters:** Cross-validation on training data
only. We use Prophet's built-in CV — slide the training window forward,
evaluate on 12-month horizon, pick lowest RMSE. Test set never touched
during this process.

**SARIMA vs Prophet — key difference:**
- SARIMA: one unified equation, mean-reverting, no holiday handling
- Prophet: three separate components, extrapolates trend, explicit holidays

---

---

# PART 4 — MODEL RESULTS: AGGREGATE SERIES
*Estimated time: 5 minutes*

---

## Section 4.1 — Train/test split and baselines

**What to say:** Before any model results mean anything, we need to
establish what we are comparing against.

**The split:** 48 months training (Feb 2011 → Jan 2015),
12 months test (Feb 2015 → Jan 2016). Strictly time-based — no shuffling,
no leakage. Every model trained on identical data, evaluated on identical
test period.

**Baseline models:**
- **Naive (persistence):** predict next month = this month's value.
  The simplest possible forecast.
- **SMA(3):** 3-month simple moving average. Slightly smoother.

These are the floor. Any statistical model that cannot beat these
has no value.

**Aggregate baseline results:**
- Naive MAPE: 8.96% — actually beats SMA here because the upward trend
  makes the most recent value the best single estimate. SMA averages in
  older, lower values and undershoots.

**[CHART 9]** `09_baseline_forecasts.png`
The side-by-side baseline forecast plot from notebook 2 Section 3.
Both series with naive and SMA overlaid on actuals.

---

## Section 4.2 — SARIMA results: aggregate series

**What to say:** SARIMA beats the baseline — but has a specific failure mode.

**Results:** RMSE $277,220 | MAE $252,147 | MAPE 6.91%
Improvement over naive: 2.05 percentage points, $102K less error/month.

**What it gets right:** The seasonal shape is correct. Early months
(Feb–Apr 2015) are very accurate at 1.7–3.4% error.

**What it gets wrong — mean reversion:** SARIMA's AR structure is
mean-reverting by design. As the forecast horizon extends, the model
gradually pulls predictions back toward the historical mean rather than
continuing the upward trend. By month 12 (Jan 2016) the error reaches
10.9%.

**Definition — mean reversion:** the tendency of a model to pull long-horizon
forecasts back toward the average of the training data, even when the
true series continues trending in one direction.

**[CHART 10]** `10_sarima_aggregate.png`
SARIMA forecast vs actual for aggregate series. The widening gap
at longer horizons should be visible.

---

## Section 4.3 — Prophet results: aggregate series

**What to say:** Prophet fixes the mean reversion problem completely.

**Results:** RMSE $209,726 | MAE $178,567 | MAPE 5.02%
Best result on aggregate. Beats SARIMA by 1.89 pp, $67K less error/month.

**Why it wins:** Piecewise linear trend with changepoint detection
extrapolates the slope forward rather than reverting to the mean.
SARIMA's month-12 error: 10.9%. Prophet's month-12 error: 1.3%.
Same month, same data, completely different horizon behavior.

**Unexpected finding — multiplicative seasonality:**
The EDA assumed additive seasonality but CV grid search found
multiplicative wins convincingly. Why? Revenue doubled from $2M to $4M
over 5 years — seasonal swings grew proportionally. A $150K seasonal
peak in 2011 became ~$300K by 2016. Additive seasonality assumes a
fixed dollar amplitude and systematically underestimates later years.

**[CHART 11]** `11_prophet_aggregate_forecast.png`
Prophet forecast vs actual for aggregate series. Show 95% CI.

**[CHART 12]** `12_prophet_aggregate_components.png`
Prophet component decomposition — trend, yearly seasonality,
quarterly seasonality, holiday effects.

---

---

# PART 5 — MODEL RESULTS: INDIVIDUAL PRODUCT-STORE
*Estimated time: 4 minutes*

---

## Section 5.1 — Why individual series are harder

**What to say:** Everything changes when you go from platform aggregate
to a single product at a single store.

**The representative series:** FOODS_3_163 @ CA_3 — selected as the
series closest to the median revenue among all 2,469 complete series.
Average monthly revenue: ~$120.

**Why it's harder:**
- Median daily sales: 2 units
- Skewness: 11.89 (extremely right-skewed)
- Any single promotion, stockout, or price change can double or halve
  a month's revenue
- The law of large numbers no longer smooths things out

**Baseline results at this level:**
- Naive MAPE: 55.29%
- SMA(3) MAPE: 45.80%
The floor is much higher here.

---

## Section 5.2 — SARIMA and Prophet: individual series

**What to say:** Both models improve substantially over baseline —
and then hit the same wall.

**SARIMA:** RMSE $54.41 | MAPE 22.22% — beats SMA by 23.6 pp (51% relative improvement)
**Prophet:** RMSE $52.80 | MAPE 24.25% — statistically tied with SARIMA

**They are tied** — SARIMA wins on MAPE, Prophet wins on RMSE.
Neither is meaningfully better.

**More importantly: they fail on the same months.**
April 2015: SARIMA 54.9%, Prophet 58.5%
May 2015: SARIMA 47.3%, Prophet 49.8%
January 2016: SARIMA 55.6%, Prophet 40.0%

**This is not a coincidence.** Two completely different model families,
same failure months, similar magnitudes. This is the signal ceiling.

**[CHART 13]** `13_signal_ceiling.png`
The side-by-side bar chart of month-by-month errors for SARIMA vs
Prophet on the representative series. Red shading on the three
shared failure months. From notebook 3b Section 13.

---

## Section 5.3 — The signal ceiling explained

**What to say:** The signal ceiling is the most important finding
of the entire project.

**What is the signal ceiling?**
The maximum accuracy a model can achieve using only historical time
series data — no matter how sophisticated the model is.

**Why do both models fail on the same months?**
Those months were driven by external events that neither model can
observe from time series history alone:
- April/May 2015: likely a price drop event. EDA confirmed r=0.553
  for price drops at the product-store level — a 10% price drop
  produces ~77% demand surge.
- January 2016: likely SNAP distribution timing or post-holiday
  restocking spike.

**The fix is not a better statistical model. It is different inputs.**
XGBoost in the next phase ingests `sell_price`, `price_change_pct`,
and `snap_TX/CA/WI` directly — the exact signals that caused these spikes.

---

---

# PART 6 — STORE-LEVEL RESULTS
*Estimated time: 4 minutes*

---

## Section 6.1 — Why model stores independently?

**What to say:** Store-level forecasts bridge the gap between platform
aggregate and individual product-store. Each store gets its own Prophet
model because stores have very different revenue baselines and growth
trajectories.

**The three stores:**
- CA_3: 17.1% of platform revenue — highest volume store
- CA_1: 12.0% — second highest
- TX_2: 10.9% — third highest

**Hyperparameter approach:** Each store gets its own CV grid search.
The aggregate winning parameters (cps=0.5, multiplicative) are not
assumed to transfer — different series, different optimal settings.

---

## Section 6.2 — CA_1 and CA_3 results

**CA_1 — 2.66% MAPE:** The best result in the entire project at any
hierarchy level. Every month within 5.1%, final month at 0.2%.
CA_1 has the smoothest, most consistent trajectory — ideal conditions
for Prophet.

**CA_3 — 6.23% MAPE:** Good result but with a consistent directional
bias — overshoots almost every month. CA_3's actual growth decelerated
slightly in the test period vs the training trend, so Prophet projects
higher than actuals. November 2015 largest miss at 11.8%.

**[CHART 14]** `14_store_ca1_forecast.png`
CA_1 Prophet forecast vs actual.

**[CHART 15]** `15_store_ca3_forecast.png`
CA_3 Prophet forecast vs actual.

---

## Section 6.3 — TX_2 and the univariate ceiling

**TX_2 — 15.31% MAPE:** Undershoots every single month. The gap
widens from 12.9% in February to 21.7% by January — the signature
of a trend slope underestimate compounding over time.

**Why it cannot be fixed with Prophet alone:**
TX_2's year-over-year growth accelerated from 8.2% during training
to 11.6% in the test period. Prophet, trained on the slower rate,
structurally undershoots. This is a fundamental limitation of
univariate forecasting — not a tuning failure.

TX_2's SNAP uplift is the strongest signal: TX FOODS showed +17.2%
revenue on SNAP days in the EDA. Prophet sees none of this. XGBoost
with `snap_TX` as an explicit feature directly encodes the demand
signal that drives TX_2's outperformance.

**[CHART 16]** `16_store_tx2_forecast.png`
TX_2 Prophet forecast vs actual. The systematic undershoot and
widening gap should be clearly visible.

---

---

# PART 7 — MASTER COMPARISON AND CONCLUSIONS
*Estimated time: 4 minutes*

---

## Section 7.1 — Master model comparison table

**What to say:** Every model, every series, one table.
Walk through it top to bottom.

**Platform aggregate revenue:**

| Model | RMSE ($) | MAE ($) | MAPE |
|---|---|---|---|
| Naive | 380,051 | 332,241 | 8.96% |
| SMA(3) | 443,310 | 396,285 | 10.71% |
| SARIMA(2,0,1)(0,1,1)[12] | 277,220 | 252,147 | 6.91% |
| **Prophet — cps=0.5, multiplicative** | **209,726** | **178,567** | **5.02%** |

**Individual product-store — FOODS_3_163 @ CA_3:**

| Model | RMSE ($) | MAE ($) | MAPE |
|---|---|---|---|
| Naive | 98.48 | 91.31 | 55.29% |
| SMA(3) | 85.66 | 77.31 | 45.80% |
| **SARIMA(0,0,1)(0,1,1)[12]** | **54.41** | **37.39** | **22.22%** |
| Prophet — cps=0.01, additive | 52.80 | 39.68 | 24.25% |

*SARIMA wins MAPE, Prophet wins RMSE — statistical tie.*

**Store level — top 3 stores:**

| Store | Revenue share | Model | RMSE ($) | MAE ($) | MAPE |
|---|---|---|---|---|---|
| **CA_1** | 12.0% | **Prophet — cps=0.01, multiplicative** | **13,104** | **11,313** | **2.66%** |
| CA_3 | 17.1% | Prophet — cps=0.01, multiplicative | 39,078 | 36,472 | 6.23% |
| TX_2 | 10.9% | Prophet — cps=0.05, additive | 56,856 | 55,343 | 15.31% |

**[CHART 17]** `17_mape_comparison_bar.png`
The bar chart from notebook 3b Section 12 — MAPE by model for
aggregate and representative series side by side.

---

## Section 7.2 — Three answers to the business question

**What to say:** Return to the original question and give concrete answers.

**Q: How many units will store X sell next month?**

At the platform level: within 5 cents per dollar (5.02% MAPE).
Accurate enough for budget planning and category-level procurement.

At the store level: 2.7–15.3% depending on the store. CA_1 and CA_3
are operationally useful. TX_2 requires external features to improve.

At the individual product-store level: within 22% on a median series.
Wide confidence intervals — useful for flagging anomalies, not for
precise unit-level ordering.

**Q: Are sudden sales spikes genuine demand or anomalies?**

The shared failure months (Apr/May 2015, Jan 2016) are confirmed
anomalies — demand spikes that no historical pattern can explain.
These are the targets for anomaly detection in the next phase.

---

## Section 7.3 — What comes next: XGBoost

**What to say:** Statistical models have hit their ceiling. The next
phase builds the ML layer.

**Three things XGBoost adds that statistical models cannot do:**

**1. External features**
- `sell_price` + `price_change_pct`: directly encodes the price signal
  that drove the shared failure months (r=0.553 at product-store level)
- `snap_CA/TX/WI`: directly encodes the SNAP calendar
  (+10–32% demand uplift confirmed in EDA)
- `is_event_day`: SuperBowl +18.9%, LaborDay +19.6%

**2. Recent momentum via lag features**
- `lag_7`, `lag_28`: sales from 1 week / 4 weeks ago
- `rolling_mean_7`, `rolling_mean_28`: smoothed recent trend
- Unlike SARIMA's AR terms — computed from recent actuals and adaptive

**3. Scale across the full hierarchy**
- One SARIMA per series caps at 2,469 complete series
- XGBoost trains one model across all 30,490 simultaneously
- Store and department encodings capture individual baselines
- Evaluated via walk-forward cross-validation to prevent leakage

**Expected targets:**

| Series | Statistical ceiling | XGBoost target |
|---|---|---|
| Aggregate | 5.02% MAPE | Marginal — Prophet already handles trend well |
| Representative | 22.22% MAPE | < 15% — price and SNAP close the gap |
| TX_2 store | 15.31% MAPE | Largest expected gain |

---

## Section 7.4 — Limitations

**What to say:** Be honest about what the models don't do.

- Revenue is derived (`units × price`) — minor gaps where price records
  are missing fill with zero, slightly understating revenue
- 8.1% complete series — SARIMA only viable for 2,469 of 30,490 series
- One representative series — the 22% MAPE is a point estimate at the
  median, not a guarantee across the full distribution
- Monthly aggregation — all statistical models here operate on monthly
  data. XGBoost targets daily, where zero-inflation makes it harder
- Training data ends April 2016 — no COVID, no e-commerce era

---

---

# CHART SAVE INSTRUCTIONS

Save all charts to:
```
outputs/presentation_charts/
```

**Complete chart list:**

| # | Filename | Source | Notes |
|---|---|---|---|
| 1 | `01_dataset_overview.png` | New | Schema diagram of 3 files |
| 2 | `02_zero_inflation.png` | EDA Section 5/22 | Units sold histogram |
| 3 | `03_revenue_by_category.png` | EDA Section 7 | Category + dept bars |
| 4 | `04_seasonal_decomposition.png` | EDA Section 10 | 4-panel decomp |
| 5 | `05_snap_effect.png` | EDA Section 12 | SNAP uplift by state |
| 6 | `06_price_elasticity.png` | EDA Section 18 | 6-panel scatter |
| 7 | `07_series_completeness.png` | EDA Section 15 | Active months histogram |
| 8 | `08_acf_pacf.png` | EDA Section 21 | 4-panel ACF/PACF |
| 9 | `09_baseline_forecasts.png` | NB2 Section 3 | Naive + SMA both series |
| 10 | `10_sarima_aggregate.png` | NB2 Section 4c | SARIMA agg forecast |
| 11 | `11_prophet_aggregate_forecast.png` | NB3 Section 3c | Prophet agg forecast |
| 12 | `12_prophet_aggregate_components.png` | NB3 Section 3c | Prophet components |
| 13 | `13_signal_ceiling.png` | NB3b Section 13 | SARIMA vs Prophet errors |
| 14 | `14_store_ca1_forecast.png` | NB3b Section 6 | CA_1 forecast |
| 15 | `15_store_ca3_forecast.png` | NB3b Section 6 | CA_3 forecast |
| 16 | `16_store_tx2_forecast.png` | NB3b Section 6 | TX_2 forecast |
| 17 | `17_mape_comparison_bar.png` | NB3b Section 12 | MAPE bar chart |

**Total: 17 charts**

Most of these already exist as outputs from your notebooks.
You will need to re-run the relevant cells with `plt.savefig()`
added before `plt.show()`, pointing to the folder above.

---

---

# TIMING GUIDE

| Part | Content | Time |
|---|---|---|
| Part 1 | The problem + dataset | 4 min |
| Part 2 | EDA (5 findings) | 7 min |
| Part 3 | Stationarity + model selection | 5 min |
| Part 4 | Aggregate results | 5 min |
| Part 5 | Individual series + signal ceiling | 4 min |
| Part 6 | Store-level results | 4 min |
| Part 7 | Comparison + conclusions + next steps | 4 min |
| **Total** | | **33 min** |

Questions from professor will likely add 5–10 minutes on top.
Be ready to explain: why SARIMA needs stationarity, what a changepoint
is in Prophet, why the signal ceiling is not fixable with more tuning,
and what walk-forward CV means.