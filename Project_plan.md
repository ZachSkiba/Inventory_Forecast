# TriXpense — Full EDA Plan + Project Roadmap

---

## What We Are Building

A time series forecasting system that predicts monthly spending — both at the **platform level** (all users aggregated) and at the **individual user level** (personalized forecasts). This maps directly to what a product data scientist at Capital One or Morningstar would build.

**The forecasting question we are answering:**
> *Given a user's past spending history, how much will they spend next month — and will they exceed their budget?*

---

## EDA Plan — `01_eda.ipynb`

EDA is not just plots. Every section must answer a specific question that justifies a modeling decision later. This is what separates a senior data scientist's notebook from a student's notebook.

---

### Section 1 — Imports and Setup ✅ Done

### Section 2 — Load Raw Data ✅ Done
- Combined fraudTrain.csv + fraudTest.csv → 1,852,394 rows, 23 columns
- Justified combining: Kaggle split is random, we need a time-based split for forecasting

### Section 3 — Clean Data ✅ Done
- Removed 9,651 fraudulent transactions (0.52% of data)
- Dropped 10 irrelevant columns (names, redundant coordinates, IDs)
- Kept 12 meaningful columns for EDA, time series, and ML phases
- Parsed dates, sorted chronologically
- Engineered `age` from `dob` — dropped raw `dob`
- Final shape: 1,842,743 rows × 12 columns

### Section 4 — Sanity Check ✅ Done
- Date range: Jan 1 2019 → Dec 31 2020 (exactly 24 months)
- 908 unique users, 14 categories, 693 merchants
- No missing values anywhere
- Amount range: $1 → $28,948 | Mean: $67.65 | Median: $47.24

### Section 5 — Category Breakdown ✅ Done
- 14 spending categories
- Top 3: grocery_pos (15.93%), shopping_pos (9.79%), gas_transport (9.57%)
- grocery_net is smallest at 2.79%
- This informs which categories to prioritize in category-level forecasting

---

### Section 6 — Transaction Amount Distribution 🔲 Next

**Question we are answering:** Are individual transaction amounts heavily skewed or full of outliers? This affects whether monthly aggregates will be stable or driven by a few extreme transactions.

- Histogram of all transaction amounts (full range)
- Histogram zoomed to 95th percentile (to see real shape without outlier distortion)
- Mean vs median comparison
- Skewness statistic
- % of transactions under $100, under $500, over $1,000

**Why it matters for modeling:** If monthly totals are dominated by a handful of $10,000+ transactions, our time series will have unstable variance (heteroskedasticity) and SARIMA may need a log transformation.

---

### Section 7 — Monthly Trend: All Users (Aggregate Series) 🔲

**Question we are answering:** Does total platform spending have a visible trend and seasonal pattern over 24 months? This is the primary justification for using SARIMA.

- Aggregate all transactions to monthly totals → 24 data points
- Line plot of monthly total spending Jan 2019 → Dec 2020
- Overlay 3-month rolling average to smooth noise
- Annotate December months to highlight holiday spending
- Print the full 24-row monthly table

**Why it matters:** If we see a clear seasonal pattern (e.g. spending peaks in December, dips in February), SARIMA's seasonal component is justified. If the series looks completely random, we need a different approach.

---

### Section 8 — Seasonal Decomposition: Aggregate Series 🔲

**Question we are answering:** Can we formally separate trend, seasonality, and residual noise from the aggregate spending series?

- Run `seasonal_decompose` with `model='additive'`, `period=12`
- Plot all 4 components: observed, trend, seasonal, residual
- Interpret each component in a markdown cell:
  - Trend: is spending growing or flat over 2019–2020?
  - Seasonality: which months are consistently high/low?
  - Residual: is there structure left, or is it white noise?

**Why it matters:** This plot is the single most important justification for SARIMA in your class write-up. It visually proves the series has decomposable structure. A professor will look for this before anything else.

---

### Section 9 — Category-Level Monthly Trends 🔲

**Question we are answering:** Do different spending categories have different seasonal patterns? Do they move together or independently?

- Aggregate monthly spending per category → 14 series × 24 months
- Line plot: all 14 categories on one chart (use colormap)
- Separate plots for top 5 categories for clarity
- Correlation heatmap between category monthly series

**Why it matters:** If categories are highly correlated, a VAR model (summer phase) makes sense. If they move independently, per-category SARIMA is the right approach. This is the justification for Phase 1 of the summer work.

---

### Section 10 — User-Level Analysis 🔲

**Question we are answering:** How consistent is individual user spending? Which users have complete 24-month histories with no gaps?

- Count how many months of data each user has
- Histogram of months-per-user distribution
- Identify users with all 24 months present (no gaps) — these are candidates for individual modeling
- Print count: how many users have complete histories?

**Why it matters:** SARIMA requires a complete time series with no gaps. We cannot model a user who only has 10 months of data. This step tells us how many users are actually modelable.

---

### Section 11 — Pick Representative User + User EDA 🔲

**Question we are answering:** Which single user best represents "typical" spending behavior for our individual-level model?

**How to pick:**
- Filter to users with all 24 months
- Among those, find the user whose monthly spending is closest to the median user (most representative, not extreme)
- Not the highest spender, not the lowest — the most typical

**User EDA plots:**
- Monthly spending trend for selected user (line plot with rolling average)
- Category breakdown for that user (horizontal bar chart)
- Seasonal decomposition for that user's series
- Compare their spending pattern to the aggregate — are they representative?

**Why it matters:** This user becomes your individual forecast subject. Their story is: "TriXpense user [ID] — here is their predicted spending for the next 6 months and whether they will exceed their budget." That is the product feature.

---

### Section 12 — Save Processed Files 🔲

Save three files to `data/processed/` — each serves a different modeling phase:

| File | Contents | Used For |
|---|---|---|
| `transactions_clean.csv` | Full 1.84M cleaned transactions | XGBoost feature engineering (summer) |
| `monthly_spending.csv` | 24-row aggregate monthly totals | SARIMA + Prophet on aggregate series |
| `monthly_user_[cc_num].csv` | 24-row monthly totals for representative user | SARIMA + Prophet on individual user |

---

## High-Level Project Roadmap

---

### Week 1 — Class Submission

| Notebook | What It Does | Key Output |
|---|---|---|
| `01_eda.ipynb` | Full EDA as above | 3 processed files, all plots, written observations |
| `02_sarima.ipynb` | SARIMA on aggregate + individual user | Forecast plots, RMSE, MAE, confidence intervals |
| `03_prophet.ipynb` | Prophet on same two series + comparison | Side-by-side model comparison, budget alert logic |

**Class deliverable:** One clean, well-documented notebook suite that reads like a professional report. Every plot has a title, axis labels, and a markdown cell interpreting what it shows and why it matters.

---

### Summer Phase 1 — Category-Level Forecasting (Weeks 1–3)

- Separate SARIMA or Prophet model per category (14 models)
- Walk-forward cross-validation — the correct way to evaluate time series (no data leakage)
- VAR model — models all categories simultaneously, captures interdependencies
- Model selection table: RMSE, MAE, MAPE across all approaches

**New notebook:** `04_category_forecasting.ipynb`

---

### Summer Phase 2 — XGBoost ML Layer (Weeks 4–6)

Build a proper ML feature table from `transactions_clean.csv`:

| Feature | Description |
|---|---|
| `lag_1`, `lag_2`, `lag_3` | User's spending 1, 2, 3 months ago |
| `rolling_mean_3` | 3-month rolling average spending |
| `rolling_mean_6` | 6-month rolling average spending |
| `rolling_std_3` | Spending volatility over 3 months |
| `month_of_year` | Captures seasonality explicitly (1–12) |
| `age` | User demographic |
| `gender` | User demographic |
| `city_pop` | Urban vs rural context |
| `pct_grocery`, `pct_entertainment`, ... | Category spending ratios per user per month |
| `num_transactions` | Transaction frequency that month |

- XGBoost regression: predict next month's spending per user
- Walk-forward cross-validation
- Compare XGBoost vs SARIMA vs Prophet in one table
- SHAP feature importance plot — which features drive predictions most?

**New notebook:** `05_xgboost_forecasting.ipynb`

---

### Summer Phase 3 — Anomaly Detection (Week 7)

- Extract residuals from best model (likely XGBoost)
- Flag months where residual exceeds 2 standard deviations → statistically unusual spending
- Isolation Forest as alternative anomaly detection approach
- Visualize flagged months on spending timeline
- Business framing: *"Unusual spike in entertainment spending detected — 2.4σ above forecast"*

**New notebook:** `06_anomaly_detection.ipynb`

---

### Summer Phase 4 — Streamlit Dashboard (Weeks 8–9)

Four pages:

1. **Forecast page** — select a user, see their 6-month spending forecast with confidence intervals and budget threshold line
2. **Category breakdown** — where is this user's money going, and what is predicted per category
3. **Anomaly flags** — timeline showing months flagged as unusual with explanation
4. **Model comparison** — SARIMA vs Prophet vs XGBoost metrics side by side

> This is what recruiters actually open. A live interactive app on your resume link is worth more than any static notebook.

---

### Summer Phase 5 — Portfolio Polish (Week 10)

- Clean GitHub repo with professional README (problem, data, methods, results, how to run)
- Write-up: 1-page explanation of methodology and business impact in plain English
- Finalized resume bullet:

> *"Developed end-to-end predictive spending analytics for 908 users using SARIMA, Prophet, and XGBoost on engineered time series features from 1.84M transactions. Applied walk-forward cross-validation, SHAP-based feature importance, anomaly detection on forecast residuals, and delivered personalized budget insights in an interactive Streamlit dashboard."*

---

## Why Every Piece Matters for Internships

| What You Built | What It Signals to a Recruiter |
|---|---|
| Aggregate + per-user SARIMA/Prophet | You understand when to use time series models and can apply them correctly |
| Walk-forward cross-validation | You know how to evaluate models without data leakage — most students get this wrong |
| XGBoost on engineered features | You can do real ML, not just plug-and-play sklearn |
| SHAP feature importance | You can explain model outputs in business terms |
| Anomaly detection | Directly maps to fraud and risk detection at Capital One |
| Streamlit dashboard | You ship things — not just notebooks |
| Clean documentation throughout | You can communicate to non-technical stakeholders |