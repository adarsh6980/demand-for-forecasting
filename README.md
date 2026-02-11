# 🛍️ Adaptive Retail Demand Forecasting System

An ML-powered demand forecasting system for retail stores — built with **XGBoost**, **drift detection**, **business rules**, and a **Streamlit dashboard**. Tested with real **Zara** product data.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-Forecasting-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📈 **XGBoost Forecasting** | 7–14 day demand predictions per SKU |
| 🛍️ **Real Zara Data** | Cleaned and integrated Zara product catalog (252 products) |
| 🌤️ **Weather Integration** | Real historical weather from Open-Meteo API |
| 🔍 **Drift Detection** | ADWIN, DDM (River) + KS-test for concept drift |
| ⚙️ **Business Rules Engine** | Capacity, budget & perishability constraints |
| 📊 **Interactive Dashboard** | Streamlit-based UI with Plotly visualizations |
| 🔄 **Continuous Learning** | Auto-retrain models when drift is detected |
| 📦 **Order Recommendations** | Constraint-aware ordering with manual overrides |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/adarsh6980/demand-for-forecasting.git
cd demand-for-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

The project includes a Zara dataset. Clean and transform it:

```bash
python src/clean_zara_data.py
```

This will:
- Clean the raw `data/zara.csv` (252 Zara products)
- Generate 180 days of daily sales time-series
- Fetch real weather data from Open-Meteo API
- Auto-create business rules in `config/business_rules.yml`

> **Alternative:** Use the UCI Online Retail II dataset instead:
> ```bash
> python src/download_real_data.py
> ```

### 3. Train Models

```bash
python src/forecasting.py
```

### 4. Launch Dashboard

```bash
streamlit run src/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
demand-for-forecasting/
├── data/
│   ├── zara.csv               # Raw Zara product catalog
│   ├── pos_data.csv           # Cleaned daily POS data
│   ├── external_data.csv      # Weather data
│   └── overrides.csv          # User order overrides
├── models/                    # Trained XGBoost models (.joblib)
├── config/
│   └── business_rules.yml     # SKU constraints (auto-generated)
├── logs/
│   └── model_events.csv       # Drift & retrain event logs
├── src/
│   ├── data_ingestion.py      # Load & validate CSVs
│   ├── feature_engineering.py # Calendar, lag, rolling, weather features
│   ├── forecasting.py         # XGBoost models + continuous learning
│   ├── drift_detection.py     # ADWIN/DDM/KS-test drift detection
│   ├── business_rules.py      # Constraint engine (YAML-driven)
│   ├── diagnostics.py         # Drift diagnostic reports
│   ├── dashboard.py           # Streamlit dashboard app
│   ├── scheduler_stub.py      # Daily automation pipeline
│   ├── clean_zara_data.py     # Zara data cleaning & transformation
│   └── download_real_data.py  # UCI/Kaggle dataset downloader
└── requirements.txt
```

---

## 📊 Dashboard Sections

### 1. Forecast & Orders
- View demand forecasts per SKU with interactive charts
- See order recommendations with business constraints applied
- Submit manual overrides with reason tracking

### 2. Business Rules
- Edit capacity, budget, perishability settings per SKU
- Save changes to YAML config in real-time

### 3. Drift & Diagnostics
- View drift alerts with severity levels
- Run on-demand drift analysis
- See diagnostic reports with actionable recommendations

### 4. Model Performance
- Track model improvement over retraining cycles
- View feature importance rankings
- Monitor prediction accuracy (MAE, R², RMSE)

---

## 🔄 Pipeline Architecture

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────┐
│  Data Ingest │───▶│ Feature Engineer  │───▶│  XGBoost     │
│  (POS + Wx)  │    │ (24 features)     │    │  Forecasting │
└──────────────┘    └───────────────────┘    └──────┬───────┘
                                                    │
                    ┌───────────────────┐           │
                    │  Drift Detection  │◀──────────┘
                    │  (ADWIN/DDM/KS)   │
                    └────────┬──────────┘
                             │ drift detected?
                    ┌────────▼──────────┐    ┌──────────────┐
                    │  Auto-Retrain     │───▶│  Business    │
                    │  (if severity>50%)│    │  Rules       │
                    └───────────────────┘    └──────┬───────┘
                                                    │
                                            ┌───────▼───────┐
                                            │  Dashboard    │
                                            │  (Streamlit)  │
                                            └───────────────┘
```

---

## ⚙️ Business Rules Schema

```yaml
skus:
  - sku: "COTTON_BLEND_BOMBER_JACKET"
    max_shelf_capacity: 120
    unit_cost: 25.99
    max_budget_per_order: 3000
    perishability_days: 120    # Fashion lifecycle
    safety_stock_days: 3
```

---

## 🔍 Drift Detection Methods

| Detector | Type | What It Detects |
|----------|------|-----------------|
| **ADWIN** | Streaming | Changes in prediction residual magnitude |
| **DDM** | Streaming | Increases in prediction error rate |
| **KS-Test** | Batch | Feature distribution shifts over time |

---

## 🛍️ Zara Data Details

The included `data/zara.csv` contains **252 real Zara products** across 5 categories:

| Category | Count | Seasonality Pattern |
|----------|-------|-------------------|
| Jackets | 140 | Peak in autumn/winter |
| Sweaters | 41 | Peak in autumn/winter |
| T-Shirts | 32 | Peak in spring/summer |
| Shoes | 31 | Mild, bimodal (spring + autumn) |
| Jeans | 8 | Stable, slight back-to-school bump |

The cleaning script (`clean_zara_data.py`) selects the top 20 products by sales volume and generates realistic daily time-series with:
- Weekly patterns (weekends +30%)
- Category-specific seasonal curves
- Holiday spikes (Black Friday, Christmas, New Year)
- Promotion effects (+40-80% during sale seasons)
- Real weather data from Open-Meteo API

---

## 📋 Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies

Key libraries:
- `xgboost` — Gradient boosting models
- `streamlit` — Interactive dashboard
- `plotly` — Visualizations
- `river` — Online drift detection (ADWIN, DDM)
- `scikit-learn` — ML utilities
- `pandas`, `numpy` — Data processing

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
