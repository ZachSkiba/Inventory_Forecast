# FULL PROJECT HANDOFF — M5 RETAIL DEMAND FORECASTING / DEMAND-TO-INVENTORY DECISION SYSTEM

You are taking over an ongoing portfolio project. Act as a **Principal Data Scientist with strong software-engineering judgment** and continue from the exact state below.

Do NOT restart the project.

Do NOT redesign the modeling methodology.

Do NOT retrain models.

Do NOT silently change frozen artifacts.

Do NOT assume that proposed code from earlier conversations was actually installed.

The project is now transitioning from **frozen ML/data science → application engineering/deployment**.

Your job is to preserve the scientific integrity of the work while turning it into a polished, recruiter-facing application.

---

# 1. PROJECT

**M5 Retail Demand Forecasting — Demand-to-Inventory Decision System**

The project uses the M5 retail dataset and combines:

* demand-regime segmentation
* model selection
* forecasting
* uncertainty quantification
* inventory policy
* service-level scenarios
* cost sensitivity
* explainability
* portfolio risk monitoring
* application-ready data packaging

Final demand regimes:

| Regime       | Forecast method   | Inventory method         | Explanation                |
| ------------ | ----------------- | ------------------------ | -------------------------- |
| Smooth       | LightGBM Tweedie  | Dynamic policy           | SHAP                       |
| Erratic      | LightGBM Tweedie  | Dynamic policy           | SHAP                       |
| Intermittent | TSB               | Dynamic policy           | Routing/policy explanation |
| Lumpy        | Historical policy | Historical demand policy | Routing/policy explanation |

The frozen production model is:

`../data/processed/models/tweedie_optimized_fold3.txt`

It is the exact frozen Fold 3 production artifact.

Fold 3 is the final untouched evaluation set.

---

# 2. CORE PRINCIPLE

The modeling system is already frozen.

Notebook 09 is an application-packaging notebook, not a modeling notebook.

The governing principle is:

**FROZEN SOURCE → VALIDATE → PACKAGE → DOCUMENT → APP**

NOT:

**FROZEN SOURCE → RECREATE → MODIFY → REPLACE**

Never fix a population mismatch by inventing new predictions.

Never silently replace missing frozen trajectories.

Never recalculate regime assignments in Notebook 09.

Never recompute SHAP.

Never retrain or tune.

Never treat scenario results as validated production results unless explicitly documented as such.

---

# 3. FROZEN POPULATION

Final Fold 3 population:

**30,490 SKUs**

Regime distribution:

| Regime       |   SKUs | Share |
| ------------ | -----: | ----: |
| Intermittent | 17,303 | 56.7% |
| Smooth       |  8,111 | 26.6% |
| Lumpy        |  4,170 | 13.7% |
| Erratic      |    906 |  3.0% |

Frozen routing thresholds:

* ADI threshold = 1.32
* CV² threshold = 0.49

Assignments are frozen in:

`../data/processed/segmentation/sku_regimes_fold3.parquet`

Do not recreate the labels from ADI/CV² in Notebook 09.

---

# 4. FROZEN MODEL

Production model:

`../data/processed/models/tweedie_optimized_fold3.txt`

Model:

* LightGBM Tweedie
* 2,000 trees
* 39 frozen features

Feature list:

`../data/processed/features/feature_cols_v2.pkl`

Fold 3 validation features:

`../data/processed/features/features_val_v2.parquet`

Rows:

11,128,850

Validation period:

2015-02-01 → 2016-01-31

Feature order MUST match `feature_cols_v2.pkl`.

---

# 5. IMPORTANT FROZEN FORECAST COVERAGE FACT

Frozen Tweedie routing population:

**9,017**

Frozen daily prediction trajectory population:

**8,863**

Missing daily trajectories:

**154**

Frozen prediction artifact:

`../data/processed/predictions/final_predictions_fold3.parquet`

Facts:

* rows = 3,234,995
* unique prediction SKUs = 8,863
* 365 rows per trajectory
* date range = 2015-02-01 → 2016-01-31

There are:

* 0 forecast-only normalized IDs
* 154 Tweedie-only normalized IDs

These 154 missing trajectories are a real frozen-source-system coverage limitation.

**DO NOT create replacement predictions for them.**

This must remain explicit in the app.

Also distinguish the related residual-monitor populations:

9,017 Tweedie

* 122 with no monitor residual data
  = 8,895 with residual data

* 32 below residual floor
  = 8,863 valid per-SKU residual distributions / frozen trajectories

Do not collapse these populations into one number.

---

# 6. NOTEBOOK 08 EXPLAINABILITY

Frozen Notebook 08 artifacts:

`../data/processed/predictions/explainability/shap_global_fold3.parquet`

`../data/processed/predictions/explainability/shap_by_regime_fold3.parquet`

`../data/processed/predictions/explainability/shap_by_department_fold3.parquet`

`../data/processed/predictions/explainability/shap_per_sku_fold3.parquet`

`../data/processed/predictions/explainability/shap_regime_comparison_fold3.parquet`

`../data/processed/predictions/explainability/representative_explainability_fold3.parquet`

`../data/processed/predictions/explainability/shap_validation_report.json`

Notebook 08 QA:

PASS

Per-SKU SHAP:

* 9,017 Tweedie SKUs
* 45,085 rows
* exactly 5 rows per SKU
* exactly 5 unique features per SKU
* ranks 1–5
* finite SHAP values
* direction labels validated
* frozen regime assignments validated
* all features belong to frozen 39-feature set

SHAP is used only for:

* Smooth
* Erratic

Intermittent and Lumpy do NOT receive SHAP.

They receive routing/policy explanations instead.

Important wording:

SHAP = predictive contribution, NOT causality.

Price SHAP = NOT price elasticity.

---

# 7. NOTEBOOK 09 PURPOSE

Notebook:

`09_app_data_prep.ipynb`

Purpose:

Convert frozen Fold 3 production artifacts into compact, versioned artifacts for Streamlit.

No:

* model training
* hyperparameter tuning
* model selection
* new holdout evaluation
* SHAP calculation
* recalibration
* replacement prediction generation
* expensive runtime simulation

The app should load compact precomputed artifacts and perform lightweight deterministic operations.

The app must clearly state:

**Forecast data as of: 2016-01-31**

This is a historical M5 decision-support demonstration, NOT a live 2026 retailer deployment.

---

# 8. NOTEBOOK 09 STATUS

## Section 0 — COMPLETE

Frozen source contract validated.

## Section 1 — COMPLETE

`app_manifest.json`

Configuration includes:

* project version = 1.0.0
* app schema = 1.0.0
* model = LightGBM Tweedie
* model version = fold3_final
* feature version = feature_cols_v2
* regime version = Syntetos-Boylan-Fold3-v1
* data as-of = 2016-01-31
* primary q80
* lead time = 7 days
* review period = 1 week
* primary horizon = 28 days
* 11 major app artifacts registered

Manifest was later finalized with:

`finalized = true`

## Section 2 — COMPLETE

`app_sku_metadata.parquet`

Rows:

30,490

Columns:

17

Includes:

* id
* item_id
* store_id
* dept_id
* cat_id
* state_id
* mean_weekly_demand
* median_weekly_demand
* zero_demand_rate
* adi
* cv2
* regime
* routing_method
* latest_available_demand
* latest_weekly_demand
* data_as_of
* price

Important distinction:

Metadata `data_as_of` is tied to the available historical training data in the earlier version, while the app-level forecast snapshot is 2016-01-31. Do not casually overwrite historical semantics.

## Section 3 — COMPLETE

`app_forecasts.parquet`

Actual:

* 3,234,995 rows
* 8,863 unique forecast SKUs
* 7-day summaries = 469,739 rows
* 28-day summaries = 2,995,694 rows
* forecast as of = 2016-01-31

Columns include:

* id
* date
* forecast_horizon_days
* item_id
* store_id
* dept_id
* cat_id
* state_id
* regime
* routing_method
* point_forecast
* q50
* q75
* q80
* q90
* q95
* q99

Daily quantiles validated as monotonic:

q50 ≤ q75 ≤ q80 ≤ q90 ≤ q95 ≤ q99

The 154 missing frozen trajectories are intentionally preserved as a gap.

## Section 4 — COMPLETE

`app_regime_forecasts.parquet`

Rows:

30,490

Regime counts:

* Smooth = 8,111
* Erratic = 906
* Intermittent = 17,303
* Lumpy = 4,170

Forecast status:

* Frozen production forecast = 26,166
* Frozen fallback forecast = 32
* Historical proxy — not ML forecast = 122
* Historical policy estimate = 4,170

Important:

Smooth + Erratic = 9,017

Of those:

* 8,863 frozen daily trajectories
* 32 frozen fallback forecasts
* 122 historical proxies

Do not describe the 122 historical proxies as ML predictions.

## Section 5 — COMPLETE

`app_inventory_policy.parquet`

Rows:

121,472

Unique SKUs:

30,368

Service levels:

* q80
* q90
* q95
* q99

IMPORTANT:

The `service_level` field is stored numerically:

* 0.80
* 0.90
* 0.95
* 0.99

The `validated_configuration` field is:

* TRUE for all 30,368 q80 rows
* FALSE for all 30,368 q90 rows
* FALSE for all 30,368 q95 rows
* FALSE for all 30,368 q99 rows

Primary validated configuration:

* 7-day lead time
* 1-week review
* q80

q90/q95/q99 = scenario alternatives.

122 SKUs have no frozen inventory policy and remain excluded from this table.

Do not recompute frozen q80 order quantities.

## Section 6 — COMPLETE

`app_explainability.parquet`

Rows:

66,558

Unique SKUs:

30,490

Columns:

14

Structure:

45,085 SHAP rows

* 17,303 Intermittent routing rows

* 4,170 Lumpy routing rows

= 66,558 rows

SHAP rows:

* Smooth / Erratic
* 5 per SKU
* frozen Notebook 08 values

Routing rows:

* Intermittent → TSB
* Lumpy → Historical policy
* 1 per SKU
* no fake SHAP

The app explainability artifact also contains a controlled `feature_description` column.

## Section 7 — COMPLETE

`app_portfolio_risk.parquet`

Rows:

30,490

Unique SKUs:

30,490

Important:

M5 does NOT contain observed current inventory.

Therefore:

* current inventory was NOT fabricated
* true current stockout risk is NOT claimed where observed inventory is unavailable

The risk table includes deterministic monitoring fields built from frozen artifacts, including recent forecast-error diagnostics and demand volatility.

A regime-change flag was discussed.

It is NOT critical to the core app because the actual current regime is frozen and no live regime evolution is being modeled. Do not invent a historical regime-change process just to populate a flag.

## Section 8 — COMPLETE

Outputs:

`app_model_results.parquet`

`app_inventory_results.parquet`

Historical model performance is packaged from existing frozen artifacts.

QA result:

* model-result rows = 21
* forecast/model rows = 9
* stability rows = 12
* inventory-result rows = 16

Selected production model:

**LightGBM / Tweedie Raw**

Regime distribution:

* Erratic = 906
* Intermittent = 17,303
* Lumpy = 4,170
* Smooth = 8,111

Service-level coverage in stored service-level artifact:

* q50 = 30,368
* q75 = 30,368
* q80 = 30,368
* q90 = 30,368
* q95 = 30,368
* q99 = 30,368

Important distinction:

The separate simulation artifact is q80 only and stores:

`service_level = 0.8`

Weekly in-stock rate:

`1 - simulated stockout rate`

Fill rate is separate and unit-based.

Lumpy WAPE:

Unavailable because `n_real_forecast = 0`

The pooled q80 modeled annual cost that was observed is:

`$1,989,158`

But that pooled artifact represented:

`29,555 SKUs`

Do not imply it represents all 30,490 SKUs.

No new simulation or model computation was performed.

## Section 9 — COMPLETE

Outputs:

`app_cost_sensitivity.parquet`

`app_policy_tradeoffs.parquet`

Both have:

24 rows

Cost sensitivity:

6 stockout-cost multipliers:

1.5, 2, 3, 4, 6, 8

4 carrying-cost rates:

15%, 20%, 25%, 30%

24 total scenarios.

Actual observed frozen dynamic-cost range:

$1,403,290 → $6,303,232

Static modeled-cost range:

$15,239,337 → $80,717,242

Naive modeled-cost range:

$24,090,816 → $128,288,958

Static-vs-dynamic head-to-head has:

24 scenarios

`static_optimal_level = q99` for all 24 tested scenarios.

Do NOT describe q99 as a universal theoretical optimum.

Use:

**best tested static configuration**

not:

“theoretical optimum.”

These results are modeled historical scenario results, NOT realized savings.

## Section 10 — COMPLETE

Purpose:

Final release QA.

All required app artifacts exist.

Passed:

* population reconciliation
* forecast coverage reconciliation
* inventory-policy reconciliation
* q80 validation
* q90/q95/q99 scenario validation
* explainability reconciliation
* portfolio-risk reconciliation
* model-result reconciliation
* inventory-result reconciliation
* cost-sensitivity reconciliation
* policy-tradeoff reconciliation
* historical snapshot validation

Selected production model:

LightGBM / Tweedie Raw

Primary inventory:

q80

Lead time:

7 days

Review:

1 week

Manifest finalized:

TRUE

Release QA:

PASS

Artifact:

`app_release_qa.json`

Section 10 final app package status:

**READY**

---

# 9. CURRENT SECTION 11 STATE

Section 11 is the final handoff summary.

The initial Section 11 code created:

`app_handoff_summary.json`

However, there was a reporting-order problem:

`app_files` was scanned BEFORE the JSON file was written.

Therefore the initial printed artifact list did not show the new JSON.

The corrected Section 11 code was changed so that:

1. it builds the handoff summary
2. writes `app_handoff_summary.json`
3. verifies the JSON can be reloaded
4. rescans the app directory AFTER writing
5. prints the final artifact inventory
6. explicitly verifies the handoff JSON exists
7. prints the absolute path

The intended final artifact list is:

* app_manifest.json
* app_release_qa.json
* app_handoff_summary.json
* app_sku_metadata.parquet
* app_forecasts.parquet
* app_regime_forecasts.parquet
* app_inventory_policy.parquet
* app_explainability.parquet
* app_portfolio_risk.parquet
* app_model_results.parquet
* app_inventory_results.parquet
* app_cost_sensitivity.parquet
* app_policy_tradeoffs.parquet

The corrected Section 11 should be run and verified before moving on.

Do NOT claim the handoff JSON exists until the corrected code actually shows it.

---

# 10. TERMINOLOGY

Use:

* frozen Fold 3
* frozen production artifact
* historical decision-support demonstration
* scenario alternative
* best tested static configuration
* predictive contribution
* routing explanation
* modeled historical cost
* historical application snapshot

Avoid:

* live
* real-world savings
* causal
* price elasticity when referring to SHAP
* validated q90/q95/q99
* current retailer performance
* new prediction when describing a frozen fallback or historical proxy
* theoretical optimum
* guaranteed savings
* production performance in 2026

---

# 11. IMPORTANT SOFTWARE-ENGINEERING GOAL

The next phase is the actual Streamlit application.

The app should NOT depend on notebooks being open.

The app should consume only the frozen application bundle.

Architecture:

```text
Notebook 01–08
      ↓
frozen modeling outputs
      ↓
Notebook 09
      ↓
validated app data contract
      ↓
Streamlit application
```

The Streamlit application should NOT:

* train LightGBM
* run SHAP
* rerun inventory simulation
* recompute service-level optimization
* recreate regime segmentation
* generate replacement forecasts
* reach into arbitrary modeling notebooks

Runtime should mainly do:

* filtering
* sorting
* lookup
* joins
* aggregation
* deterministic scenario selection
* plotting
* display formatting

---

# 12. DESIRED STREAMLIT EXPERIENCE

This is the user's primary portfolio project and should be polished enough for recruiters.

The app should immediately communicate:

**M5 Retail Demand Forecasting — Demand-to-Inventory Decision System**

Important headline facts:

* 30,490 SKUs
* 4 demand regimes
* 8,863 frozen daily forecast trajectories
* q80 primary inventory configuration
* 7-day lead time
* 1-week review
* historical snapshot: 2016-01-31

Suggested navigation:

1. Overview
2. SKU Explorer
3. Forecast Explorer
4. Inventory Planning
5. Portfolio Risk
6. Explainability
7. Policy Trade-offs
8. Methodology / Data Contract

The interface should feel like a decision-support product, not a notebook.

---

# 13. PROPOSED SOFTWARE ARCHITECTURE

Target structure:

```text
m5-retail-demand/
│
├── app/
│   ├── streamlit_app.py
│   ├── config.py
│   │
│   ├── data/
│   │   ├── loader.py
│   │   ├── validators.py
│   │   └── selectors.py
│   │
│   ├── views/
│   │   ├── overview.py
│   │   ├── sku_explorer.py
│   │   ├── forecasting.py
│   │   ├── inventory.py
│   │   ├── risk.py
│   │   ├── explainability.py
│   │   └── policy_tradeoffs.py
│   │
│   └── utils/
│       ├── formatting.py
│       └── constants.py
│
├── data/
│   └── processed/
│       └── predictions/
│           └── app/
│
├── tests/
│   ├── test_loaders.py
│   ├── test_validators.py
│   └── test_selectors.py
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
├── README.md
└── ...
```

BUT:

**Do not assume this directory already exists.**

First inspect the real repository and adapt to its actual structure.

Do not create duplicate architectures.

---

# 14. DATA-LAYER DESIGN

Use a clean data access layer.

Conceptually:

```text
streamlit_app.py
       ↓
data.loader
       ↓
validated cached DataFrames
       ↓
selectors / business-display logic
       ↓
views
```

Do not scatter:

```python
pd.read_parquet(...)
```

throughout every page.

The manifest should be read first.

Validate:

* manifest finalized
* artifact presence
* model/configuration consistency
* expected schema

Then load application data.

Use caching where appropriate.

---

# 15. LARGE-ARTIFACT CONCERN

`app_forecasts.parquet` is:

3,234,995 rows

and is much larger than the other app artifacts.

Before deciding repository/deployment architecture, inspect actual file sizes.

Do NOT assume that all app data should simply be committed directly into Git.

The application must remain free for the user and recruiter-facing.

---

# 16. DEPLOYMENT GOAL

User wants the app:

* completely free
* publicly accessible
* always available for recruiters
* portfolio-quality
* stable
* not dependent on local machine

Important practical constraint:

Streamlit Community Cloud is free, but apps can sleep after inactivity, so it is not literal 24/7 always-on hosting.

Render free services also sleep when idle.

Therefore, investigate a truly free or always-free host suitable for a small Dockerized Streamlit service.

One previously considered option:

**Oracle Cloud Always Free VM**

Potential architecture:

```text
GitHub
   ↓
Docker
   ↓
Oracle Cloud Always Free VM
   ↓
Caddy / HTTPS
   ↓
Streamlit
   ↓
public recruiter URL
```

Do NOT blindly implement cloud deployment yet.

First inspect the repo, app artifacts, dependency files, and current app state.

Because cloud pricing/free-tier policies can change, verify current deployment constraints from official sources before making claims.

---

# 17. SECURITY / DEPLOYMENT PRINCIPLES

No secrets should be committed.

No database is required for the first release unless there is a compelling reason.

No model serving API is necessary.

No authentication is required for the public portfolio demo unless needed by the chosen host.

The app should be read-only.

The application data is historical/public-project data.

Use HTTPS.

Use a health check if the host supports it.

Use a Docker container so the deployment is reproducible.

---

# 18. RECRUITER UX REQUIREMENTS

The app should make the DS/ML work obvious without requiring the recruiter to understand the entire notebook sequence.

A recruiter should be able to:

* identify the production model
* inspect a SKU
* view its demand regime
* see the forecast
* inspect inventory recommendations
* compare service levels
* inspect SHAP explanations
* view portfolio risk
* inspect policy trade-offs
* understand the historical evaluation scope
* understand limitations

The app must also clearly expose limitations rather than hiding them.

Examples:

For a SKU with no frozen trajectory:

> No frozen daily production trajectory is available for this SKU. The application does not generate a replacement forecast.

For current inventory:

> Observed current inventory is unavailable in the historical M5 dataset.

For cost savings:

> Modeled historical scenario result — not realized business savings.

For SHAP:

> SHAP values represent predictive contribution, not causal effect.

---

# 19. DO NOT DO THIS

Do not:

* retrain models in Streamlit
* load the full raw M5 dataset unless absolutely necessary
* recreate the regime segmentation
* fabricate current inventory
* generate replacement forecasts for the 154 missing trajectories
* claim all 30,490 SKUs have daily ML forecasts
* call Lumpy historical-policy estimates ML forecasts
* call q90/q95/q99 validated Fold 3 configurations
* claim modeled cost savings are realized
* call weekly in-stock rate conventional unit fill rate
* call SHAP causal
* call price SHAP elasticity
* silently change frozen artifacts
* write a giant one-file Streamlit script if the repo supports clean modules

---

# 20. IMMEDIATE NEXT STEP

Before writing the Streamlit application, inspect the actual repository.

Generate a PowerShell discovery report showing:

* project root
* top-level folders/files
* existing Streamlit files
* existing `app/` files
* `requirements.txt`
* `pyproject.toml`
* Docker files
* `.streamlit` files
* Git status
* Git LFS status
* current Python environment
* installed versions of:

  * streamlit
  * pandas
  * pyarrow
  * numpy
  * plotly
* all Notebook 09 app artifacts
* exact sizes of all app artifacts
* files over 50 MB
* files over 100 MB
* candidate Streamlit entrypoints

Do NOT ask the user to guess paths that can be discovered with PowerShell.

---

# 21. WORKING STYLE

Act like a Principal Data Scientist / Staff-level engineer.

Be conservative.

When something is uncertain:

1. inspect the actual file/schema
2. diagnose
3. then modify

Never guess.

When fixing code:

* provide the full replacement section
* keep it reasonably compact
* avoid unnecessary abstraction
* preserve validated frozen logic

When evaluating output:

* distinguish:

  * true bug
  * frozen-source limitation
  * schema mismatch
  * earlier assumption error

Do not change methodology merely to satisfy an assertion.

Assertions should validate known facts, not invent them.

---

# 22. CURRENT STATE

Notebook 09:

* Section 0 ✅
* Section 1 ✅
* Section 2 ✅
* Section 3 ✅
* Section 4 ✅
* Section 5 ✅
* Section 6 ✅
* Section 7 ✅
* Section 8 ✅
* Section 9 ✅
* Section 10 ✅
* Section 11 code corrected but **needs final execution verification**

Current app artifacts already produced:

```text
app_manifest.json
app_sku_metadata.parquet
app_forecasts.parquet
app_regime_forecasts.parquet
app_inventory_policy.parquet
app_explainability.parquet
app_portfolio_risk.parquet
app_model_results.parquet
app_inventory_results.parquet
app_cost_sensitivity.parquet
app_policy_tradeoffs.parquet
app_release_qa.json
```

Expected final handoff artifact:

```text
app_handoff_summary.json
```

It must be physically verified before saying it exists.

---

# 23. BIG PICTURE

This project is now beyond the main modeling phase.

The scientific/modeling layer is frozen.

The next phase is to turn the existing artifacts into a **real software product**.

Desired final lifecycle:

```text
M5 data
   ↓
EDA
   ↓
baselines
   ↓
Prophet / SARIMA
   ↓
XGBoost
   ↓
LightGBM / Tweedie
   ↓
regime segmentation
   ↓
uncertainty
   ↓
inventory simulation
   ↓
Fold 3 freeze
   ↓
SHAP
   ↓
Notebook 09 app bundle
   ↓
Streamlit application
   ↓
Docker
   ↓
free public hosting
   ↓
recruiter-facing portfolio app
```

The application is the final presentation layer for the entire project.

The goal is not merely to make the app work.

The goal is to make it look and behave like a **small, thoughtfully engineered analytics product** while remaining scientifically honest about what the frozen historical M5 results actually represent.

Start by inspecting the repository and current app state. Do not jump directly into writing the UI until the actual repo structure and artifact sizes are known.
