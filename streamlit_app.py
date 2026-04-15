import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="Metal Leaching Prediction App",
    layout="centered"
)

st.title("🔮 Metal Leaching Efficiency Predictor")

st.write(
    """
    Predict recovery of **Li, Co, Mn, Ni** using trained ML models.
    """
)

# --------------------------------------------------------------
# MODEL LOADING
# --------------------------------------------------------------
@st.cache_resource
def load_models():

    models = {}

    base_path = os.path.join(os.getcwd(), "models")

    if not os.path.exists(base_path):
        st.error(f"❌ Models folder not found at: {base_path}")
        return models

    files = os.listdir(base_path)

    metals = ["Li", "Co", "Mn", "Ni"]
    labels = ["withcat", "nocat"]

    missing_files = []

    for metal in metals:
        for label in labels:

            filename = f"best_tuned_{label}_{metal}.pkl"
            filepath = os.path.join(base_path, filename)

            if os.path.exists(filepath):
                try:
                    with open(filepath, "rb") as f:
                        models[f"{label}_{metal}"] = pickle.load(f)

                except ModuleNotFoundError as e:
                    st.error(
                        f"""
                        ❌ Failed to load {filename}

                        Missing dependency in environment:
                        {e}

                        👉 Fix: add required library to requirements.txt
                        """
                    )

                except Exception as e:
                    st.error(f"❌ Error loading {filename}: {e}")

            else:
                missing_files.append(filename)

    if missing_files:
        st.warning("⚠️ Missing model files:")
        for m in missing_files:
            st.write("-", m)

    return models


models = load_models()

# --------------------------------------------------------------
# CHECK IF MODELS LOADED
# --------------------------------------------------------------
if not models:
    st.stop()

# --------------------------------------------------------------
# INPUTS
# --------------------------------------------------------------
st.subheader("Input Parameters")

Li_feed = st.number_input("Li in feed %", 0.0, 10.0, 5.0)
Co_feed = st.number_input("Co in feed %", 0.0, 60.0, 20.0)
Mn_feed = st.number_input("Mn in feed %", 0.0, 20.0, 5.0)
Ni_feed = st.number_input("Ni in feed %", 0.0, 40.0, 10.0)

acid = st.number_input("Leaching agent concentration (M)", 0.1, 6.0, 1.5)
reduct = st.number_input("Reducing agent concentration %", 0.0, 25.0, 1.0)

time = st.number_input("Leaching time (min)", 0, 1080, 60)
temp = st.number_input("Temperature (°C)", 20, 100, 70)

# --------------------------------------------------------------
# PREDICTION
# --------------------------------------------------------------
if st.button("Predict"):

    X = pd.DataFrame([{
        "Li in feed %": Li_feed,
        "Co in feed %": Co_feed,
        "Mn in feed %": Mn_feed,
        "Ni in feed %": Ni_feed,
        "Concentration, M": acid,
        "Concentration %": reduct,
        "Time,min": time,
        "Temperature, C": temp
    }])

    results = {}

    for metal in ["Li", "Co", "Mn", "Ni"]:

        model_key = f"withcat_{metal}"

        if model_key in models:
            try:
                pred = models[model_key].predict(X)[0]
                results[metal] = pred
            except Exception as e:
                st.error(f"Prediction failed for {metal}: {e}")
        else:
            st.warning(f"Model missing: {model_key}")

    # ----------------------------------------------------------
    # OUTPUT
    # ----------------------------------------------------------
    if results:
        st.subheader("Predicted Leaching Efficiency")

        for metal, value in results.items():
            st.metric(f"{metal} %", round(value, 2))
