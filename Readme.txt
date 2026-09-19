# Air Quality Prediction System

An interactive Machine Learning web application designed to predict PM2.5 air pollution concentration using multi-site environmental and meteorological measurements.

##  Features
- **Real-Time Predictions:** Instant PM2.5 forecasting using an optimized XGBoost Regressor model.
- **Interactive UI:** Built with Streamlit, featuring live two-way synchronization between number inputs and sliders.
- **Dynamic Visualizations:** Integrated Plotly gauge charts with color-coded safety thresholds.

##  Tech Stack
- **Python**
- **Scikit-Learn & XGBoost** (Machine Learning & Imputation)
- **Streamlit** (Web Framework)
- **Plotly** (Data Visualization)

##  Project Structure
```text
air_quality_project/
│
├── models/
│   └── xgboost_optimized_pipeline.pkl
├── app.py
├── cleaned_air_quality.csv
└── requirements.txt