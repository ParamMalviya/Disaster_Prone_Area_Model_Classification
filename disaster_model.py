import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import os

# -------------------------------
# 🌍 Page Configuration
# -------------------------------
st.set_page_config(page_title="🌍 Disaster Risk Classifier", layout="wide")
st.title("🌍 Disaster-Prone Area Classification App")
st.markdown("Predict the **risk level** of a region based on geographical and disaster-related factors.")

# -------------------------------
# 📦 Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "disaster_model.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()
if model is None:
    st.error("❌ Model file not found. Please ensure `disaster_model.pkl` exists.")
    st.stop()

# -------------------------------
# 🗺️ Map Generator
# -------------------------------
def generate_risk_map(lat, lon, state, disaster, prediction, damage_scale):
    risk_color = "red" if prediction == "High" else "orange" if prediction == "Medium" else "green"
    map_obj = folium.Map(location=[lat, lon], zoom_start=6, tiles="CartoDB dark_matter")

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(
            f"<b>State:</b> {state}<br><b>Disaster:</b> {disaster}<br><b>Risk:</b> {prediction}<br><b>Damage Scale:</b> {damage_scale}",
            max_width=300
        ),
        icon=folium.Icon(color=risk_color, icon="info-sign")
    ).add_to(map_obj)

    # Optional: Add predefined high-risk zones
    high_risk_states = {
        "Bihar": [25.9, 85.1],
        "Assam": [26.2, 91.7],
        "Odisha": [20.9, 85.1],
        "Uttarakhand": [30.1, 79.3],
        "Tamil Nadu": [11.1, 78.7]
    }

    for state_name, coords in high_risk_states.items():
        folium.Circle(
            location=coords,
            radius=50000,
            color="red",
            fill=True,
            fill_opacity=0.3,
            popup=f"High Risk: {state_name}"
        ).add_to(map_obj)

    return map_obj

# -------------------------------
# 📥 Input Form
# -------------------------------
st.header("📥 Input Region and Disaster Details")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("🗺️ Select State", [
            "Andhra Pradesh", "Assam", "Bihar", "Gujarat", "Karnataka",
            "Kerala", "Maharashtra", "Odisha", "Tamil Nadu", "Uttarakhand", "West Bengal"
        ])
        disaster_type = st.selectbox("🌪️ Disaster Type", [
            "Flood", "Earthquake", "Cyclone", "Landslide", "Fire", "Drought", "Tsunami"
        ])
        elevation = st.number_input("⛰️ Elevation (meters)", min_value=0, max_value=9000, value=100)
        disaster_score = st.slider("📊 Disaster History Score", 0, 10, 5)
        pop_density = st.number_input("👥 Population Density (per km²)", min_value=0, max_value=20000, value=5000)
        latitude = st.number_input("📍 Latitude", min_value=5.0, max_value=40.0, value=22.5)

    with col2:
        urban_level = st.slider("🏙️ Urbanization Level", 0.0, 1.0, 0.5, step=0.05)
        houses_affected = st.number_input("🏚️ Houses Affected (last 5 years)", 0, 10000, 100)
        deaths = st.number_input("☠️ Human Deaths (last 5 years)", 0, 1000, 10)
        longitude = st.number_input("📍 Longitude", 65.0, 100.0, 78.9)

    submitted = st.form_submit_button("🔍 Predict Risk")

# -------------------------------
# 🧠 Prediction & Map Display
# -------------------------------
if submitted:
    try:
        # 🎯 Feature Engineering
        risk_index = disaster_score * 0.5 + pop_density * 0.3 + urban_level * 0.2
        damage_scale = houses_affected + (deaths * 10)

        input_data = pd.DataFrame([{
            "Disaster_History_Score": disaster_score,
            "Population_Density": pop_density,
            "Urbanization_Level": urban_level,
            "House_Affected": houses_affected,
            "Human_Death": deaths,
            "Risk_Index": risk_index,
            "Damage_Scale": damage_scale,
            "Elevation": elevation
        }])

        model_features = getattr(model, "feature_names_in_", input_data.columns.tolist())
        missing = [col for col in model_features if col not in input_data.columns]
        if missing:
            raise ValueError(f"Missing input features: {missing}")

        # 🔮 Prediction
        prediction = model.predict(input_data)[0]

        # Store results
        st.session_state["prediction"] = prediction
        st.session_state["map_data"] = {
            "lat": latitude,
            "lon": longitude,
            "state": state,
            "disaster": disaster_type,
            "damage_scale": damage_scale
        }

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

# -------------------------------
# ✅ Show Prediction and Map
# -------------------------------
if "prediction" in st.session_state and "map_data" in st.session_state:
    prediction = st.session_state["prediction"]
    map_data = st.session_state["map_data"]

    st.success(f"✅ **Predicted Risk Level: `{prediction}`**")
    st.header("🗺️ Map: Predicted Location")

    with st.spinner("Generating map..."):
        map_result = generate_risk_map(
            map_data["lat"], map_data["lon"], map_data["state"],
            map_data["disaster"], prediction, map_data["damage_scale"]
        )
        st_folium(map_result, width=700, height=500)
