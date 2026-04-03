# TriXpense Predictive Analytics Project Plan

## One Week — Class Submission
**Theme:** Predicting monthly user spending for TriXpense using time series forecasting

### Day 1–2: Data + EDA
- Download Kaggle Credit Card Transactions or Personal Finance dataset  
- Map to TriXpense schema: `user_id`, `date`, `amount`, `category`, `recurring_flag`  
- Aggregate to monthly totals  
- Decomposition plot — trend, seasonality, residual (looks great, easy to do)  
- Basic EDA:  
  - Spending distribution  
  - Category breakdown  
  - Monthly trends  

### Day 3–4: Modeling
- **SARIMA** — fit, tune, forecast 3–6 months ahead  
- **Prophet** — run on same series, same forecast horizon  
- Compare: RMSE, MAE side by side  
- Plot actual vs predicted for both models  

### Day 5: Business Insights + Write-Up
- Confidence intervals on forecast  
- Flag months where predicted spending exceeds budget threshold  
- 1–2 sentence business insight per finding  
- Clean up notebook with markdown explanations  
- Frame everything as TriXpense predictive analytics  

**Deliverable:** Clean Jupyter notebook. Solid grade, solid foundation.

---

## Summer — Full Portfolio Project
**Goal:** Tier 1 internship-ready project  

### Phase 1: Expand the Modeling (2–3 weeks)
- Category-level forecasting — separate SARIMA/Prophet model per category (Food, Transport, Entertainment, etc.)  
- Walk-forward cross-validation — proper time series evaluation  
- VAR model — capture correlations between categories simultaneously  
- Compare all approaches in a model selection table with metrics  

### Phase 2: Add the ML Layer (2–3 weeks)
- Feature engineering:  
  - Lag features (1, 2, 3 months)  
  - Rolling averages (3-month, 6-month)  
  - Month-of-year, spending ratios  
- XGBoost regression on engineered features to predict next month's spending  
- Compare XGBoost vs SARIMA vs Prophet — show which wins and why  
- Feature importance plot — shows what drives spending predictions  

### Phase 3: Anomaly Detection (1 week)
- Extract residuals from your best forecast model  
- Flag statistically unusual months using Z-score or Isolation Forest on residuals  
- Visualize anomalies on spending timeline  
- Business framing: "Unusual spike in discretionary spending detected in Month X"  

### Phase 4: Dashboard (1–2 weeks)
- Build in **Streamlit** (fastest) or **Plotly Dash**  
- Pages:  
  - Monthly forecast with confidence intervals  
  - Category-level breakdown + predictions  
  - Anomaly flags  
  - Budget threshold alerts  

> This is what recruiters actually open and interact with

### Phase 5: Portfolio Polish (1 week)
- GitHub repo with clean README  
- Write-up explaining problem, data, methodology, results, business impact  
- Resume bullet finalized  

---

## Full Project Timeline

| Phase | Timeline | Deliverable |
|-------|---------|-------------|
| Class baseline | Week 1 | Notebook: SARIMA + Prophet + decomposition |
| Category forecasting + CV | Summer Week 1–3 | Expanded notebook |
| XGBoost ML layer | Summer Week 4–6 | Full forecasting pipeline |
| Anomaly detection | Summer Week 7 | Anomaly module |
| Dashboard | Summer Week 8–9 | Streamlit app |
| Portfolio polish | Summer Week 10 | GitHub + write-up |

---

## Resume Bullet (Final Form)
> "Developed end-to-end predictive spending analytics for TriXpense using SARIMA, Prophet, and XGBoost on engineered time series features. Applied walk-forward cross-validation, anomaly detection on forecast residuals, and visualized actionable budget insights in an interactive Streamlit dashboard."

---

## Why This Works for Capital One / Morningstar
- **Time series rigor:** SARIMA, Prophet, VAR, walk-forward CV  
- **ML depth:** XGBoost on engineered features, feature importance  
- **Anomaly detection:** Directly maps to fraud/risk detection roles  
- **Business framing:** Every model output tied to a budget insight  
- **Dashboard:** Tangible artifact recruiters can interact with
