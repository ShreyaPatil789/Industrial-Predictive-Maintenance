import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np
from config import MODEL_PATH, PREDICTION_THRESHOLD

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ AI-Based Predictive Maintenance Dashboard")
st.markdown("Real-Time Machine Health Monitoring & Risk Intelligence")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# Simulation Toggle
# -----------------------------
simulate = st.toggle("🔄 Enable Live Simulation", value=False)

# -----------------------------
# Initialize History Storage
# -----------------------------
if "risk_history" not in st.session_state:
    st.session_state.risk_history = []

# -----------------------------
# Sensor Inputs
# -----------------------------
st.sidebar.header("🔧 Sensor Inputs")

if not simulate:
    air_temp = st.sidebar.slider("Air Temperature [K]", 290, 330, 300)
    process_temp = st.sidebar.slider("Process Temperature [K]", 300, 350, 310)
    rot_speed = st.sidebar.slider("Rotational Speed [rpm]", 1000, 3000, 1500)
    torque = st.sidebar.slider("Torque [Nm]", 10, 80, 40)
    tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 300, 100)
    machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])
else:
    air_temp = np.random.randint(290, 330)
    process_temp = np.random.randint(300, 350)
    rot_speed = np.random.randint(1000, 3000)
    torque = np.random.randint(10, 80)
    tool_wear = np.random.randint(0, 300)
    machine_type = np.random.choice(["L", "M", "H"])

# Encode Machine Type
type_L = 1 if machine_type == "L" else 0
type_M = 1 if machine_type == "M" else 0

# -----------------------------
# Prepare Data
# -----------------------------
input_data = pd.DataFrame([{
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rot_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "Type_L": type_L,
    "Type_M": type_M
}])

# -----------------------------
# Prediction
# -----------------------------
probability = float(model.predict_proba(input_data)[0][1])
prediction = int(probability >= PREDICTION_THRESHOLD)

# Store last 20 history values
st.session_state.risk_history.append(probability * 100)
if len(st.session_state.risk_history) > 20:
    st.session_state.risk_history.pop(0)

# -----------------------------
# Risk Classification
# -----------------------------
if probability < 0.2:
    risk_level = "🟢 Low Risk"
elif probability < 0.5:
    risk_level = "🟡 Medium Risk"
else:
    risk_level = "🔴 High Risk"

# -----------------------------
# Display Main Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Failure Probability", f"{round(probability*100,2)} %")
col2.metric("Risk Level", risk_level)

if prediction == 1:
    col3.error("⚠️ Maintenance Required")
else:
    col3.success("✅ Machine Healthy")

# -----------------------------
# Business Impact
# -----------------------------
st.subheader("💰 Business Impact Analysis")

FAILURE_COST = 50000

if probability >= 0.7:
    estimated_cost = FAILURE_COST
elif probability >= 0.3:
    estimated_cost = FAILURE_COST * 0.5
else:
    estimated_cost = 0

st.metric("Estimated Risk Exposure (₹)", int(estimated_cost))

# -----------------------------
# Maintenance Recommendation Engine
# -----------------------------
st.subheader("🔧 Maintenance Recommendation")

if probability >= 0.7:
    st.error("Immediate inspection required. High failure probability detected.")
elif probability >= 0.4:
    st.warning("Schedule preventive maintenance within 48 hours.")
elif torque > 70:
    st.warning("High torque detected. Check mechanical stress levels.")
elif tool_wear > 250:
    st.warning("Tool wear critical. Consider replacement soon.")
else:
    st.success("Machine operating within safe parameters.")

# -----------------------------
# Risk Trend Chart
# -----------------------------
st.subheader("📈 Risk Trend (Last 20 Readings)")

history_df = pd.DataFrame({
    "Reading": range(1, len(st.session_state.risk_history)+1),
    "Failure Probability (%)": st.session_state.risk_history
})

st.line_chart(history_df.set_index("Reading"))

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("📊 Model Feature Importance")

importance = pd.DataFrame({
    "Feature": input_data.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance.set_index("Feature"))

# -----------------------------
# Auto Refresh
# -----------------------------
if simulate:
    time.sleep(2)
    st.rerun()