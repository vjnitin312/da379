# AirWatch India: AQI Inspection and Prediction

Term Project — DA379, Trimester 9
B.Sc. (Honours) Data Science and Artificial Intelligence, IIT Guwahati

**Author:** Nitin Vijay N (23035010571) · n.nitin@op.iitg.ac.in

An end-to-end data science pipeline for inspecting and predicting Air Quality Index (AQI) across 26 Indian cities (2015–2020), covering data cleaning, exploratory analysis, feature engineering, time-series forecasting, machine learning prediction, and an interactive Streamlit dashboard.

---

## 📊 Project Overview

- **Dataset:** Daily pollutant concentrations for 26 Indian cities across 21 states, sourced from the Central Pollution Control Board (CPCB) via [data.gov.in](https://data.gov.in)
- **Records:** 29,531 raw → **29,530 cleaned records, 0 missing values**
- **Key result:** Random Forest regressor achieves **R² = 0.925** for AQI prediction and **83.8% accuracy** for AQI category classification
- **Live demo:** [YouTube — 10 min walkthrough](https://youtu.be/y57es9DNrsM)

---

## 🗂️ Repository Structure

```
da379/
├── data/
│   ├── raw/                        # city_day.csv, stations.csv
│   └── processed/                  # cleaned & feature-engineered CSVs, coordinate lookups
├── scripts/
│   ├── phase1_cleaning.py          # Data cleaning + CPCB AQI recomputation
│   ├── phase2_eda.py               # Exploratory data analysis (7 charts)
│   ├── phase3_feature_engineering.py  # Date, lag, rolling, season features
│   ├── phase4_geo_visualization.py # Folium map — all 26 cities
│   ├── phase5_time_series.py       # Holt-Winters forecasting (top 10 cities)
│   └── phase6_ml_modeling.py       # ML training, prediction & evaluation
├── dashboard/
│   └── app.py                      # Streamlit dashboard (4 tabs)
├── outputs/
│   ├── figures/                    # 13 static analysis charts (PNG)
│   ├── maps/                       # 2 interactive Folium maps (HTML)
│   └── tables/                     # Predictions, forecasts, model metrics (CSV)
├── report/
│   └── da379_report_nitin_vijay_23035010571/  # IEEE-format LaTeX report + PDF
├── ppt/
│   └── da379_ppt_nitin_vijay_23035010571/  # Project in 9 slide view
├── video/
│   └── da379_video_nitin_vijay_23035010571/  # 10 minute project demonstration
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data processing | pandas, numpy |
| Machine learning | scikit-learn (Random Forest, Decision Tree, Linear/Logistic Regression) |
| Time-series forecasting | statsmodels (Holt-Winters Exponential Smoothing) |
| Visualization | matplotlib, seaborn, plotly |
| Geographic maps | folium |
| Dashboard | Streamlit |

---

## 🚀 Getting Started

### 1. Clone and set up environment
```bash
git clone https://github.com/vjnitin312/da379.git
cd da379
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the pipeline (in order)
```bash
python scripts/phase1_cleaning.py
python scripts/phase2_eda.py
python scripts/phase3_feature_engineering.py
python scripts/phase4_geo_visualization.py
python scripts/phase5_time_series.py
python scripts/phase6_ml_modeling.py
```

### 3. Launch the dashboard
```bash
streamlit run dashboard/app.py
```
Then open `http://localhost:8501` in your browser.

---

## 📈 Pipeline Summary

| Phase | Description | Output |
|---|---|---|
| 1. Data Cleaning | Impute missing pollutants (city+month median fallback); recompute 4,680/4,681 missing AQI values via official CPCB sub-index formula | `city_day_cleaned.csv` |
| 2. EDA | National/seasonal trends, city & state comparisons, pollutant correlations | 7 charts |
| 3. Feature Engineering | Date parts, Indian seasons, lag (1/7-day) & rolling (7/30-day) AQI features | `city_day_features.csv` |
| 4. Geo-Visualization | Interactive map — all 26 cities, colored/sized by avg AQI | `india_aqi_map.html` |
| 5. Time-Series Forecasting | Seasonal decomposition + Holt-Winters 6-month forecast for 10 most-polluted cities (≥24 months history) | Forecast table, 2 figures |
| 6. ML Prediction & Evaluation | Regression (AQI value) + Classification (AQI category); Random Forest wins both tasks | Predictions, metrics, 4 figures |
| 7. Dashboard | 4-tab Streamlit app: Overview, City Inspection, Predict AQI, Map View | `dashboard/app.py` |

---

## 🏆 Key Findings

- **Most polluted city:** Ahmedabad (avg AQI 392.6, "Very Poor")
- **Least polluted city:** Aizawl (avg AQI 34.8, "Good")
- **Seasonality:** November peaks at 234.9 AQI (winter temperature inversion); August lowest at 110.0 (monsoon washout)
- **Top AQI correlates:** CO (r=0.67), PM2.5 (r=0.66), NO2 (r=0.53) — combustion-related pollutants dominate
- **Best regression model:** Random Forest — R²=0.925, MAE=16.33, RMSE=35.37
- **Best classification model:** Random Forest — Accuracy=83.8%, F1=0.837

---

## 📄 Report

Full IEEE-format report (4 pages + references) available in [`report/`](./report), covering methodology, CPCB formula derivation, Holt-Winters equations, results, discussion, and limitations.

## 🎥 Video Presentation

10-minute walkthrough covering problem statement, code, live dashboard demo, and evaluation metrics: [Watch on YouTube](https://youtu.be/y57es9DNrsM)

---

## 🔮 Future Work

- Real-time CPCB station API integration for live inspection beyond 2015–2020
- Meteorological covariates (temperature, wind, humidity) as exogenous regressors
- SARIMA / hybrid statistical-ML comparison against Holt-Winters
- Cloud deployment of the Streamlit dashboard for public access

---

## 📜 License & Acknowledgements

Dataset sourced from the Central Pollution Control Board (CPCB) via [data.gov.in](https://data.gov.in). Built as part of the B.Sc. (Honours) Data Science and AI program at IIT Guwahati.

AI-assisted tools were used to support development. All analysis, code, and interpretations were reviewed and understood by the author.
