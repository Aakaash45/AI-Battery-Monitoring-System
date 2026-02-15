import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh
import os

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="AI Battery Monitoring", layout="wide")

# ---------------- AUTO REFRESH (8 sec for smooth UI) ----------------
st_autorefresh(interval=8000, limit=None, key="refresh")

st.title("🔋 AI-Based Battery Monitoring System")

# ---------------- SAFE CSV LOADING ----------------
file_path = os.path.join(os.getcwd(), "battery_data.csv")

if not os.path.exists(file_path):
    st.error("❌ CSV file not found! Make sure 'battery_data.csv' is in same folder.")
    st.stop()

data = pd.read_csv(file_path)

# ---------------- ANOMALY DETECTION ----------------
features = data[["Voltage", "Temperature", "Current"]]

model = IsolationForest(contamination=0.2, random_state=42)
model.fit(features)

data["Anomaly"] = model.predict(features)

# ---------------- HEALTH DEGRADATION ----------------
data["Health"] = 100 - (data.index * 2)

latest_row = data.iloc[-1]

# ---------------- ANOMALY COUNT ----------------
anomaly_count = (data["Anomaly"] == -1).sum()

# ---------------- STATUS LOGIC ----------------
if latest_row["Anomaly"] == -1:
    status = "🔴 ANOMALY DETECTED"
elif latest_row["Health"] < 60:
    status = "🟡 AGING"
else:
    status = "🟢 NORMAL"

# ---------------- METRICS ----------------
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Voltage (V)", round(latest_row["Voltage"], 2))
col2.metric("Temperature (°C)", round(latest_row["Temperature"], 2))
col3.metric("Battery Health (%)", f"{latest_row['Health']}%")
col4.metric("Remaining Cycles (RUL)", int(latest_row["Health"] * 2))
col5.metric("Total Anomalies", anomaly_count)

st.subheader(f"System Status: {status}")

st.markdown("---")

# ---------------- VOLTAGE GRAPH (Theme Friendly + Anomaly Highlight) ----------------
st.subheader("Voltage Trend with Anomaly Detection")

fig1, ax1 = plt.subplots()

ax1.plot(data["Voltage"], linewidth=2)

# Highlight anomalies in red
anomalies = data[data["Anomaly"] == -1]
ax1.scatter(anomalies.index, anomalies["Voltage"],
            color="red", s=100, label="Anomaly")

ax1.set_xlabel("Time Index")
ax1.set_ylabel("Voltage (V)")
ax1.legend()

st.pyplot(fig1)

# ---------------- TEMPERATURE GRAPH ----------------
st.subheader("Temperature Trend")
st.line_chart(data["Temperature"])

# ---------------- CURRENT GRAPH ----------------
st.subheader("Current Trend")
st.line_chart(data["Current"])

# ---------------- BATTERY HEALTH TREND ----------------
st.subheader("Battery Health Degradation Trend")
st.line_chart(data["Health"])

st.markdown("---")

# ---------------- DOWNLOAD REPORT ----------------
csv = data.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Battery Report",
    data=csv,
    file_name="battery_report.csv",
    mime="text/csv",
)

# ---------------- CSV TABLE AT BOTTOM ----------------
st.subheader("Anomaly Detection Results")
st.dataframe(data)

# ---------------- PROJECT DESCRIPTION ----------------
st.markdown("---")
st.subheader("📘 Project Description")
st.write("""
This project implements an Adaptive AI-Based Battery Fingerprinting System.
It uses Isolation Forest for anomaly detection and simulates battery health
degradation over time. The system predicts Remaining Useful Life (RUL)
and provides real-time monitoring using a Streamlit dashboard.
""")

st.success("AI Monitoring System Running Successfully 🚀")





