import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --------------------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------------------
st.set_page_config(page_title="Metal Leaching Prediction App", layout="centered")

st.title("🔮 Metal Leaching Efficiency Predictor")
st.write("""
This app uses machine learning models trained on your leaching dataset to predict the 
recovery percentages of **Li, Co, Mn, Ni**.  
Models automatically preprocess numeric + categorical values, so **raw inputs are fine**.
""")

# --------------------------------------------------------------
# Model Loader (ABSOLUTE PATH + DEBUG)
# --------------------------------------------------------------
@st.cache_resource
def load_models():
    models_loaded = {}
    
    base_path = os.path.join(os.getcwd(), "models")

    st.write("📁 **Model directory detected at:**", base_path)

    if not os.path.exists(base_path):
        st.error("❌ ERROR: 'models' folder does not exist. Upload models to /models.")
        return models_loaded

    st.write("📄 **Files found in /models:**", os.listdir(base_path))

    for metal in ["Li", "Co", "Mn", "Ni"]:
        for label in ["withcat", "nocat"]:
            filename = f"best_tuned_{label}_{metal}.pkl"
            filepath = os.path.join(base_path, filename)

            st.write(f"Checking for: `{filename}`")

            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    models_loaded[f"{label}_{metal}"] = pickle.load(f)
                st.success(f"✅ Loaded: {filename}")
            else:
                st.warning(f"⚠️ Missing model: {filename}")

    return models_loaded

models = load_models()


st.title("Lithium-Ion Battery Leaching Predictor")

st.write(
"Predict Li, Co, Mn, and Ni leaching efficiency using machine learning."
)

# ============================================================
# INPUTS
# ============================================================

Li_feed = st.number_input("Li in feed %",0.0,10.0,5.0)
Co_feed = st.number_input("Co in feed %",0.0,60.0,20.0)
Mn_feed = st.number_input("Mn in feed %",0.0,20.0,5.0)
Ni_feed = st.number_input("Ni in feed %",0.0,40.0,10.0)

acid = st.number_input("Leaching agent concentration (M)",0.1,6.0,1.5)

reduct = st.number_input("Reducing agent concentration %",0.0,25.0,1.0)

time = st.number_input("Leaching time (min)",0,1080,60)

temp = st.number_input("Temperature (°C)",20,100,70)

# ============================================================
# PREDICT
# ============================================================

if st.button("Predict"):

    X = pd.DataFrame({

        "Li in feed %":[Li_feed],
        "Co in feed %":[Co_feed],
        "Mn in feed %":[Mn_feed],
        "Ni in feed %":[Ni_feed],
        "Concentration, M":[acid],
        "Concentration %":[reduct],
        "Time,min":[time],
        "Temperature, C":[temp]

    })

    Li_pred = models["Li"].predict(X)[0]
    Co_pred = models["Co"].predict(X)[0]
    Mn_pred = models["Mn"].predict(X)[0]
    Ni_pred = models["Ni"].predict(X)[0]

    st.subheader("Predicted Leaching Efficiency")

    st.metric("Li %",round(Li_pred,2))
    st.metric("Co %",round(Co_pred,2))
    st.metric("Mn %",round(Mn_pred,2))
    st.metric("Ni %",round(Ni_pred,2))
        )
