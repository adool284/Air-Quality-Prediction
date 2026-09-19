import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="Air Quality Prediction AI",
    page_icon="🌍",
    layout="wide"
)

# =========================================================
# Custom CSS & Professional Styling / Animations
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #ffffff;
    }

    .hero {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero p {
        font-size: 1.2rem;
        opacity: 0.9;
    }

    .stButton > button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 50px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4);
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 198, 255, 0.6);
        background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(15, 32, 39, 0.85);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Load Optimized Model Pipeline
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "xgboost_optimized_pipeline.pkl"

if not model_path.exists():
    model_path = BASE_DIR / "models" / "random_forest_pipeline.pkl"

if model_path.exists():
    loaded_data = joblib.load(model_path)
    if isinstance(loaded_data, tuple):
        model, imputer, features_list = loaded_data
        is_pipeline = False
    else:
        model = loaded_data
        is_pipeline = True
else:
    st.error("Error: Model file not found in models directory!")
    st.stop()

# =========================================================
# Hero Header
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>🌫️ Air Quality Prediction System</h1>
        <p>✨ Predict PM2.5 pollutant concentration instantly using advanced Machine Learning models 🚀</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("📊 Model Insights")
    st.write("This application utilizes an optimized regression pipeline trained on multi-site environmental data.")
    st.divider()
    st.metric("R² Score", "0.9464")
    st.metric("RMSE", "18.73")
    st.divider()
    st.info("💡 Adjust sliders or type values directly for real-time live synchronization!")

# =========================================================
# Feature Configurations
# =========================================================
FEATURES = {
    "PM10": {"default": 100.0, "min": 0.0, "max": 1000.0, "step": 1.0},
    "SO2": {"default": 10.0, "min": 0.0, "max": 500.0, "step": 1.0},
    "NO2": {"default": 40.0, "min": 0.0, "max": 300.0, "step": 1.0},
    "CO": {"default": 800.0, "min": 0.0, "max": 10000.0, "step": 10.0},
    "O3": {"default": 50.0, "min": 0.0, "max": 1000.0, "step": 1.0},
    "TEMP": {"default": 20.0, "min": -20.0, "max": 45.0, "step": 0.1},
    "PRES": {"default": 1010.0, "min": 980.0, "max": 1050.0, "step": 0.1},
    "DEWP": {"default": 10.0, "min": -40.0, "max": 30.0, "step": 0.1},
    "RAIN": {"default": 0.0, "min": 0.0, "max": 70.0, "step": 0.1},
    "WSPM": {"default": 2.0, "min": 0.0, "max": 15.0, "step": 0.1},
}

# =========================================================
# Session State Initialization (Live 2-Way Sync)
# =========================================================
for feature, cfg in FEATURES.items():
    if feature not in st.session_state:
        st.session_state[feature] = cfg["default"]
    if f"{feature}_num" not in st.session_state:
        st.session_state[f"{feature}_num"] = cfg["default"]
    if f"{feature}_slider" not in st.session_state:
        st.session_state[f"{feature}_slider"] = cfg["default"]


def sync_from_num(feature):
    value = st.session_state[f"{feature}_num"]
    st.session_state[feature] = value
    st.session_state[f"{feature}_slider"] = value


def sync_from_slider(feature):
    value = st.session_state[f"{feature}_slider"]
    st.session_state[feature] = value
    st.session_state[f"{feature}_num"] = value


# =========================================================
# Input Layout (2 Columns)
# =========================================================
st.subheader("🌍 Environmental & Weather Inputs")
st.write("Use the sliders or input numbers manually for live updates 🎛️:")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 🧪 Pollutants")
    for feature in ["PM10", "SO2", "NO2", "CO", "O3"]:
        cfg = FEATURES[feature]
        st.number_input(
            feature,
            min_value=cfg["min"], max_value=cfg["max"], step=cfg["step"],
            key=f"{feature}_num", on_change=sync_from_num, args=(feature,)
        )
        st.slider(
            f"{feature} range",
            min_value=cfg["min"], max_value=cfg["max"], step=cfg["step"],
            key=f"{feature}_slider", on_change=sync_from_slider, args=(feature,),
            label_visibility="collapsed"
        )

with col2:
    st.markdown("### 🌡️ Weather Conditions")
    weather_labels = {
        "TEMP": "Temperature",
        "PRES": "Pressure",
        "DEWP": "Dew Point",
        "RAIN": "Rain",
        "WSPM": "Wind Speed"
    }
    for feature, label in weather_labels.items():
        cfg = FEATURES[feature]
        st.number_input(
            label,
            min_value=cfg["min"], max_value=cfg["max"], step=cfg["step"],
            key=f"{feature}_num", on_change=sync_from_num, args=(feature,)
        )
        st.slider(
            f"{label} range",
            min_value=cfg["min"], max_value=cfg["max"], step=cfg["step"],
            key=f"{feature}_slider", on_change=sync_from_slider, args=(feature,),
            label_visibility="collapsed"
        )

# =========================================================
# Prediction Button
# =========================================================
st.markdown("---")
b_col1, b_col2, b_col3 = st.columns([1, 2, 1])

with b_col2:
    predict_button = st.button("🔍 Run Prediction Now", use_container_width=True)

if predict_button:
    values = {feature: st.session_state[feature] for feature in FEATURES}

    values["hour"] = 12
    values["month"] = 6
    values["dayofweek"] = 2
    values["is_weekend"] = 0
    values["season"] = 1
    values["wd_encoded"] = 4

    input_df = pd.DataFrame([values])

    if not is_pipeline:
        input_df = input_df[features_list]
        input_processed = imputer.transform(input_df)
        prediction = model.predict(input_processed)[0]
    else:
        prediction = model.predict(input_df)[0]

    if prediction <= 35:
        level = "Low (Good) 🟢"
        message = "🟢 PM2.5 concentration is relatively low. Air quality is safe."
    elif prediction <= 75:
        level = "Moderate 🟡"
        message = "🟡 PM2.5 concentration is moderate. Sensitive individuals should exercise caution."
    else:
        level = "High (Unhealthy) 🔴"
        message = "🔴 PM2.5 concentration is high! Air quality is unhealthy, consider wearing masks."

    st.markdown("---")
    st.subheader("🎯 Prediction Results")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"suffix": " µg/m³", "font": {"size": 40, "color": "white"}},
            title={"text": "Predicted PM2.5 Level", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 200], "tickfont": {"color": "white"}},
                "bar": {"color": "#00c6ff"},
                "steps": [
                    {"range": [0, 35], "color": "rgba(40, 167, 69, 0.4)"},
                    {"range": [35, 75], "color": "rgba(255, 193, 7, 0.4)"},
                    {"range": [75, 200], "color": "rgba(220, 53, 69, 0.4)"}
                ],
                "threshold": {"line": {"color": "yellow", "width": 4}, "thickness": 0.75, "value": prediction}
            }
        )
    )
    gauge.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        margin=dict(l=30, r=30, t=50, b=30)
    )

    st.plotly_chart(gauge, use_container_width=True)

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.metric("Predicted PM2.5 Value", f"{prediction:.2f} µg/m³")
    with r_col2:
        st.metric("Air Quality Status", level)

    st.info(message)
    st.balloons()

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("✨ Air Quality Prediction App • Machine Learning System • 2026")
