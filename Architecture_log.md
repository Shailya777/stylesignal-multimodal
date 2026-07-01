# Architecture Log: TrendSight XAI

**Date:** June-July 2026

---

# Phase 1: Foundation & Data Ingestion

## Infrastructure
Established the project directory structure, separating:
- Raw data
- Processed pipelines
- Orchestration logic

## Data Pipeline
Created the following datasets:
- `time_series_targets.csv` — Time-series forecasting targets
- `cnn_visual_scores.csv` — CNN visual scoring pipeline output

## Initial Data Strategy
- Initially utilized `tfds.load` for the H&M Personalized Fashion Recommendations dataset.
- Subsequently migrated to local storage processing for the H&M Kaggle dataset to bypass Google Colab bandwidth bottlenecks.

---

# Phase 2: Multimodal Architecture Design

## Core Logic
Designed a multimodal pipeline that fused:
- Deep learning features (CNN-extracted visual trend scores)
- Tabular metadata including:
  - Article information
  - Department
  - Color
  - Additional product attributes

---

# Phase 3: Tabular Feature Engineering & Scaling Challenges

## High-Cardinality Resolution
Identified a dimensionality crisis caused by high-cardinality categorical variables, such as:

- `prod_name` (>8,000 unique values)

## Implementation
Engineered a hybrid encoding pipeline consisting of:

- **OneHotEncoder** for low-cardinality features
- **TargetEncoder (`ce.TargetEncoder`)** with smoothing for high-cardinality features

This approach avoided the memory explosion associated with traditional one-hot encoding.

## Data Integrity Fix — *The Bounded Skeleton*
Resolved the **Survivorship Bias** issue where implicit zero-sales months were absent.

Implemented a **Lifecycle-Aware Imputation** pipeline that:

- Added zero-demand rows only within each article's actual shelf life (`min(date)` to `max(date)`).
- Prevented artificial extension beyond the product lifecycle.
- Enabled the model to properly learn what unsuccessful products looked like.

---

# Phase 4: Model Experimentation & The Regression Trap

## Model Selection
Initiated experimentation using:

- `XGBRegressor`
- `RandomizedSearchCV`

## Objective Pivot 1
Attempted to mitigate severe zero inflation and target skew:

- **Median:** 5
- **Maximum:** 2,519

Applied:

- `TransformedTargetRegressor`
- `np.log1p`

to compress the long-tailed target distribution.

## Failure Analysis
Model evaluation produced:

- **R²:** 0.25

Post-mortem analysis showed the model reached a hard prediction ceiling, consistently underestimating highly viral products.

## Reason for Failure
The regression model lacked essential exogenous economic variables, including:

- Price
- Markdown history
- Inventory depth

As a result, the model optimized toward the geometric mean rather than true business demand, making accurate unit-level sales prediction mathematically infeasible.

---

# Phase 5: Architectural Hurdle Pivot (Hurdle Cascade)

## Architectural Pivot
Transitioned from a single regressor to a **Two-Stage Hurdle Model (Classification–Regression Cascade).**

## Design

### Stage 1 — Gatekeeper
An `XGBClassifier` predicts whether a product is likely to become viral using a **50-unit sales threshold**.

### Stage 2 — Dual Specialist Regressors

- **Regressor A:** Baseline demand predictor
- **Regressor B:** Viral-product specialist

## Gatekeeper Performance

Successfully optimized the classifier:

- **ROC-AUC:** 0.865
- **Decision Threshold:** 0.30

The optimized threshold maximized recall for high-performing (viral) products.

---

# Phase 6: Final Pre-Mortem & Project Decommissioning

## Regressor Validation Failure

### Baseline Regressor

- **MAE:** 7.30
- **Mean Target:** 7.96

### Viral Specialist

- **MAE:** 89.16
- **Mean Target:** 183.97

---

# Project Termination Criteria

## Fundamental Signal Limitation
The available features (visual aesthetics and categorical metadata) lacked the economic signals required for reliable unit-level demand forecasting.

## Diminishing Returns
Despite architectural improvements, the specialist regressors continued to exhibit error levels too high for trustworthy business analytics.

## Integrity Constraint
The project was intentionally terminated rather than deploying a model that would effectively **hallucinate sales forecasts**, avoiding downstream merchandising and inventory decisions based on unreliable predictions.

---

# Verdict

**Project decommissioned on July 1, 2026.**

The final architecture demonstrated that:

- Visual features can successfully classify a product's commercial potential (**Gatekeeper success**).
- Visual and categorical features alone cannot replace transactional economic variables for accurate regression-based demand forecasting.
- Reliable sales prediction fundamentally requires economic and operational business data.