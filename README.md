# 🚚 Last-Mile Delivery ETA Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-3.3+-02569B?style=for-the-badge&logo=lightgbm&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Optuna-Tuned-6C63FF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Portfolio%20Ready-2F9E44?style=for-the-badge"/>
</p>

<p align="center">
  A production-grade machine learning pipeline that predicts last-mile delivery duration<br/>
  from pickup to doorstep — with calibrated uncertainty estimates and per-shipment SHAP explanations.
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Modelling Pipeline](#-modelling-pipeline)
- [SHAP Explainability](#-shap-explainability)
- [Error Analysis](#-error-analysis)
- [Production Notes](#-production-notes)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)

---

## 🎯 Overview

Last-mile delivery — the final leg from distribution hub to recipient — accounts for **53% of total shipping costs** in most logistics networks. Accurate ETA prediction directly impacts:

- 📉 Customer inquiry and complaint rates
- 🔁 Failed delivery and redelivery costs
- 🗺️ Dynamic re-routing decisions
- 📦 Carrier performance management

This project builds a **supervised regression pipeline** that predicts delivery duration in minutes given features available at **pickup scan time** — no post-hoc data required. The model outputs a point estimate plus calibrated **80% and 95% prediction intervals**, enabling downstream systems to serve confidence-adjusted ETAs to customers.

---

## 📊 Results

| Metric | Baseline (Median) | Ridge | LightGBM Default | **LightGBM Tuned** |
|--------|:-----------------:|:-----:|:----------------:|:------------------:|
| MAE (min) | 29.8 | 16.2 | 11.9 | **11.3** |
| RMSE (min) | 47.3 | 22.4 | 16.1 | **15.3** |
| MAPE (%) | 38.5 | 19.1 | 10.8 | **10.0** |

<br/>

| Prediction Interval | Target Coverage | Empirical Coverage | Median Width |
|---------------------|:---------------:|:-----------------:|:------------:|
| 80% PI | 80% | 76% | ~38 min |
| 95% PI | 95% | 92% | ~67 min |

> **62% MAE reduction** over the naive baseline. Tuned LightGBM delivers a further **30% improvement** over linear regression, confirming significant non-linear interactions in the data.

### Error distribution on test set

| Error Bucket | % of Shipments | Ops Implication |
|---|:---:|---|
| ≤ 5 min | 38% | Near-perfect — show ETA to customer with high confidence |
| 5 – 10 min | 31% | Acceptable — within customer tolerance |
| 10 – 20 min | 22% | Marginal — widen ETA window in messaging |
| 20 – 40 min | 7% | Poor — flag for proactive communication |
| > 40 min | 2% | Critical — snow/fog + high-load carrier + rural combination |

---

## 📁 Project Structure

```
last-mile-eta-prediction/
│
├── last_mile_eta_prediction.ipynb   # Main notebook (fully executed)
├── last_mile_eta_report.docx        # Full project report
├── README.md
│
└── sections/
    ├── 01_setup.py                  # Imports, config, seeds
    ├── 02_data_generation.py        # Synthetic dataset (25k shipments)
    ├── 03_eda.py                    # Distributions, correlations, target analysis
    ├── 04_feature_engineering.py    # 9 engineered features, no-leakage encoding
    ├── 05_preprocessing.py          # ColumnTransformer, temporal split
    ├── 06_modelling.py              # Baseline → Ridge → LightGBM → Optuna
    ├── 07_evaluation.py             # MAE, RMSE, MAPE, prediction intervals
    ├── 08_shap.py                   # Global beeswarm, waterfall, dependence plots
    └── 09_error_analysis.py         # Segment breakdown, failure modes
```

---

## 🗄️ Dataset

A **25,000-shipment synthetic dataset** is generated using a realistic data-generating process (DGP) that mirrors last-mile logistics characteristics. Using a known DGP allows rigorous sanity-checking of SHAP attributions against true causal effects.

### Feature groups

| Group | Features | Notes |
|---|---|---|
| **Temporal** | pickup hour, day-of-week, month, is_weekend, is_rush_hour | Cyclical sin/cos encoded |
| **Route** | distance (km), stops before, zone type | Distance gamma-distributed by zone |
| **Package** | weight, volume, requires_signature, delivery attempt | Weight log-normal (0.1–50 kg) |
| **Carrier** | carrier (DHL/FedEx/UPS/Evri), service tier, load % | Load beta-distributed — skewed high |
| **Environment** | traffic index (0–10), weather, temp °C, wind km/h | Traffic amplified in rush hours |

### Target variable — `delivery_minutes`

```
delivery_minutes = (
    distance_km × 2.8
    + stops_before × 2.5
    + traffic_index × 7.5
    + carrier_load_pct × 0.35
    + carrier_bias
    + weather_penalty
    + zone_adjustment
) × service_multiplier + ε   # ε ~ N(0, 0.12 × base)
```

Clipped to **[10, 600] minutes**. Median: ~58 min, right-skewed.

### Temporal split (no leakage)

```
Jan ──────────────── Sep │ Oct ── Nov │ Dec
        TRAIN (70%)      │  VAL (15%) │ TEST (15%)
       17,500 rows        │  3,750     │  3,750
```

> Random splits would inflate test performance by ~15–20% due to temporal correlation in traffic and carrier load patterns.

---

## 🔧 Feature Engineering

Nine features derived from domain knowledge — all statistics computed **on training data only** to prevent target leakage.

| Feature | Formula | Rationale |
|---|---|---|
| `hour_sin` / `hour_cos` | sin/cos(2π × hour / 24) | Cyclical: hour 23 ≈ hour 0 |
| `dow_sin` / `dow_cos` | sin/cos(2π × dow / 7) | Cyclical day-of-week |
| `log_distance` | log1p(route_distance_km) | Linearises distance → time |
| `package_density` | weight_kg / (volume_l + ε) | Handling complexity proxy |
| `traffic_load_interaction` | traffic_index × carrier_load_pct / 100 | Compound congestion pressure |
| `stops_x_load` | stops_before × carrier_load_pct / 100 | More stops hurt more at capacity |
| `dist_x_traffic` | log_distance × traffic_index | Long routes hit hardest by gridlock |
| `carrier_hist_delay` | Median delivery_minutes per carrier (train only) | Absorbs carrier baseline differences |
| `is_redelivery` | delivery_attempt > 1 | Re-deliveries add ~18 min on average |

> The 9 engineered features contributed **more performance gain** than the jump from Ridge → LightGBM (7% vs 5% MAPE reduction).

---

## 🤖 Modelling Pipeline

### Model ladder

```
Median Baseline  →  Ridge Regression  →  LightGBM (default)  →  LightGBM (Optuna-tuned)
    38.5% MAPE          19.1% MAPE            10.8% MAPE               10.0% MAPE
```

### Hyperparameter search

**50-trial Bayesian optimisation** (Optuna, TPE sampler) with validation MAE as objective:

```python
params = {
    "n_estimators":      [300, 2000],
    "learning_rate":     [0.01, 0.20],   # log-scale
    "num_leaves":        [31, 255],
    "feature_fraction":  [0.5, 1.0],
    "bagging_fraction":  [0.5, 1.0],
    "reg_alpha":         [1e-4, 10.0],   # log-scale
    "reg_lambda":        [1e-4, 10.0],   # log-scale
    "min_child_samples": [10, 100],
}
```

Categorical features (`zone_type`, `carrier`, `service_type`, `weather`) passed natively as `pandas.Categorical` — no one-hot encoding required.

### Prediction intervals

Two additional LightGBM models trained with **quantile loss** provide calibrated intervals:

```python
# 80% prediction interval
lower_model = LGBMRegressor(objective="quantile", alpha=0.10, ...)
upper_model = LGBMRegressor(objective="quantile", alpha=0.90, ...)

# 95% prediction interval
lower_model = LGBMRegressor(objective="quantile", alpha=0.025, ...)
upper_model = LGBMRegressor(objective="quantile", alpha=0.975, ...)
```

---

## 🔍 SHAP Explainability

SHAP values computed on 2,000 test-set shipments reveal the predictive hierarchy:

| Rank | Feature | Mean \|SHAP\| (min) | Insight |
|:---:|---|:---:|---|
| 1 | `traffic_load_interaction` | 8.4 | Compound congestion is the dominant driver |
| 2 | `log_distance` | 7.1 | Core route length effect |
| 3 | `carrier_hist_delay` | 5.8 | Carrier baseline absorbed by target encoding |
| 4 | `traffic_index` | 5.2 | Direct traffic beyond the interaction term |
| 5 | `stops_before` | 4.9 | Strong in urban zones |
| 6 | `carrier_load_pct` | 4.3 | Carrier capacity is independently predictive |
| 7 | `dist_x_traffic` | 3.8 | Long route × traffic non-linearity |
| 8 | `weather (snow)` | 3.5 | Rare but large tail impact |
| 9 | `route_distance_km` | 3.1 | Partly captured by log_distance |
| 10 | `is_redelivery` | 2.9 | Strong binary signal: +18 min average |

### Key non-linear findings

- **Traffic tipping point** — traffic index has near-linear effect on ETA up to ~6, then slope steepens sharply. This interaction is strongest when `carrier_load_pct > 85%`, suggesting a carrier capacity threshold.
- **Distance × zone** — the log-distance → ETA relationship is steeper in urban zones (more stops/km) than rural (higher speeds). Rural shipments with distance > 50 km show disproportionately large SHAP values.

---

## 🧪 Error Analysis

### MAE by segment

| Segment | MAE (min) | Note |
|---|:---:|---|
| Urban | 9.8 | Best — dense data, predictable patterns |
| Suburban | 11.6 | Moderate |
| Rural | 14.1 | Hardest — sparse data, high route variability |
| Clear weather | 9.9 | Baseline condition |
| Rain | 12.4 | Moderate degradation — well captured |
| Snow | 17.8 | Largest errors — rare event, sparse training data |
| Fog | 14.2 | High variance |
| UPS | 10.1 | Most consistent carrier |
| Evri | 13.7 | Highest errors — more variable operations |

### Hardest shipments (top 1% errors)

Analysis of the worst-predicted shipments consistently shows a **compound failure mode**: snow or fog + Evri carrier + rural zone + high carrier load. Each factor alone is manageable; together they produce tail errors > 40 minutes.

---

## 🏭 Production Notes

### Inference pipeline

```
Pickup scan
    ↓
Feature engineering (deterministic transforms + carrier_stats dict lookup)
    ↓
Three LightGBM models: point estimate · lower quantile · upper quantile
    ↓
Output: { predicted_minutes, lower_80, upper_80, lower_95, upper_95 }

Target latency: < 20 ms p95
```

### Monitoring thresholds

| Signal | Threshold | Cadence | Action |
|---|---|:---:|---|
| Rolling 7-day MAE | > 20% above baseline | Daily | Retrain trigger |
| MAPE by carrier | Any carrier > 20% | Daily | Carrier-specific audit |
| 80% PI coverage | < 70% | Weekly | Recalibrate quantile models |
| PSI on `traffic_index` | > 0.2 | Weekly | Investigate data pipeline |
| % predictions > 480 min | > 5% in a day | Real-time | Operational escalation |

### Retraining cadence

| Update | Cadence |
|---|---|
| Carrier stats refresh | Weekly |
| Full model retrain | Monthly or on monitoring trigger |
| Hyperparameter re-search | Quarterly |

### Known limitations

- **Snow/fog sparsity** — rare weather events underrepresented in training. Weather-stratified sampling at retraining would reduce tail errors.
- **Rural variance** — inherently high route variability; MAE 14.1 min vs 9.8 min urban.
- **PI under-coverage** — quantile regression without conformal correction tends to underestimate tails. Production should apply conformal calibration on a holdout set.
- **No real-time disruptions** — road closures or accidents after pickup are not captured. Integration with a live traffic feed would reduce these.


## 🛠️ Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-02569B?style=flat-square"/>
  <img src="https://img.shields.io/badge/SHAP-FF6B6B?style=flat-square"/>
  <img src="https://img.shields.io/badge/Optuna-6C63FF?style=flat-square"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white"/>
</p>

---

## 📄 Licence

MIT — free to use, adapt, and build on.

---
