import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Li Battery Extraction Predictor",
    page_icon="🔋",
    layout="wide",
)

METALS = ["Li", "Co", "Mn", "Ni"]
METAL_COLORS = {"Li": "#4e9af1", "Co": "#e05c5c", "Mn": "#7bc67e", "Ni": "#f0a500"}

# Expected model filenames
MODEL_FILES = {
    metal: {
        "withcat":    f"models/best_tuned_withcat_{metal}.pkl",
        "withoutcat": f"models/best_tuned_withoutcat_{metal}.pkl",
    }
    for metal in METALS
}

# Features used by each model type
CAT_FEATURES = ["Leaching agent", "Type of reducing agent"]

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    models = {}
    for metal in METALS:
        models[metal] = {}
        for variant, fname in MODEL_FILES[metal].items():
            if os.path.exists(fname):
                with open(fname, "rb") as f:
                    models[metal][variant] = pickle.load(f)
            else:
                models[metal][variant] = None
    return models

models = load_all_models()

# ── Check which models are available ─────────────────────────────────────────
available = {
    metal: {v: m is not None for v, m in models[metal].items()}
    for metal in METALS
}
any_loaded = any(available[m][v] for m in METALS for v in ["withcat", "withoutcat"])

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔋 Li Battery Extraction Predictor")
st.markdown(
    "Predict **extraction efficiency (%)** for Li, Co, Mn, and Ni from spent "
    "lithium-ion batteries. Runs both *with-categorical* and *without-categorical* "
    "model variants side by side."
)

# Model availability status
with st.expander("📦 Model file status", expanded=not any_loaded):
    cols = st.columns(len(METALS))
    for i, metal in enumerate(METALS):
        with cols[i]:
            st.markdown(f"**{metal}**")
            for variant in ["withcat", "withoutcat"]:
                icon = "✅" if available[metal][variant] else "❌"
                st.markdown(f"{icon} `{MODEL_FILES[metal][variant]}`")
    if not any_loaded:
        st.error("No model files found. Place `.pkl` files in the same directory as `app.py`.")
        st.stop()

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("inputs"):
    st.subheader("📋 Feed Composition")
    c1, c2, c3, c4 = st.columns(4)
    li_feed = c1.number_input("Li in feed (%)",  0.0, 100.0, 7.0,  0.1)
    co_feed = c2.number_input("Co in feed (%)",  0.0, 100.0, 20.0, 0.1)
    mn_feed = c3.number_input("Mn in feed (%)",  0.0, 100.0, 10.0, 0.1)
    ni_feed = c4.number_input("Ni in feed (%)",  0.0, 100.0, 10.0, 0.1)

    st.subheader("⚗️ Leaching Conditions")
    c5, c6, c7, c8 = st.columns(4)
    leaching_agent = c5.selectbox("Leaching agent", [
        "H2SO4", "HCl", "HNO3", "H3PO4",
        "Citric acid", "Acetic acid", "Oxalic acid",
        "Tartaric acid", "DL-malic acid", "Ascorbic acid", "Succinic acid",
    ])
    leaching_conc  = c6.number_input("Concentration (M)",         0.0, 20.0,  2.0, 0.1)
    reducing_agent = c7.selectbox("Reducing agent", [
        "None", "H2O2", "Glucose", "Ascorbic acid",
        "Sucrose", "Starch", "Na2SO3", "Fe", "Al",
    ])
    reducing_conc  = c8.number_input("Reducing agent conc. (%)", 0.0, 100.0, 4.0, 0.5)

    c9, c10, _ = st.columns([1, 1, 2])
    time_min = c9.number_input("Time (min)",        0.0, 600.0, 60.0, 5.0)
    temp_c   = c10.number_input("Temperature (°C)", 0.0, 200.0, 80.0, 5.0)

    submitted = st.form_submit_button(
        "🔮 Predict All Metals", use_container_width=True, type="primary"
    )

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    row_withcat = {
        "Li in feed %":           li_feed,
        "Co in feed %":           co_feed,
        "Mn in feed %":           mn_feed,
        "Ni in feed %":           ni_feed,
        "Leaching agent":         leaching_agent,
        "Concentration, M":       leaching_conc,
        "Type of reducing agent": reducing_agent,
        "Concentration %":        reducing_conc,
        "Time,min":               time_min,
        "Temperature, C":         temp_c,
    }
    row_withoutcat = {k: v for k, v in row_withcat.items() if k not in CAT_FEATURES}

    df_withcat    = pd.DataFrame([row_withcat])
    df_withoutcat = pd.DataFrame([row_withoutcat])

    results = {}
    for metal in METALS:
        results[metal] = {}
        for variant, df in [("withcat", df_withcat), ("withoutcat", df_withoutcat)]:
            m = models[metal][variant]
            if m is not None:
                try:
                    pred = float(np.clip(m.predict(df)[0], 0, 100))
                    results[metal][variant] = pred
                except Exception as e:
                    results[metal][variant] = f"Error: {e}"
            else:
                results[metal][variant] = None

    st.divider()
    st.subheader("📊 Predicted Extraction Efficiencies")

    # ── Per-metal cards ───────────────────────────────────────────────────────
    cols = st.columns(len(METALS))
    for i, metal in enumerate(METALS):
        with cols[i]:
            color = METAL_COLORS[metal]
            st.markdown(
                f"<h3 style='color:{color}; text-align:center'>{metal}</h3>",
                unsafe_allow_html=True,
            )
            for variant in ["withcat", "withoutcat"]:
                val = results[metal][variant]
                label = "With categorical" if variant == "withcat" else "Without categorical"
                if val is None:
                    st.markdown(f"**{label}:** *(model not found)*")
                elif isinstance(val, str):
                    st.markdown(f"**{label}:** ⚠️ {val}")
                else:
                    st.metric(label=label, value=f"{val:.1f}%")
                    st.progress(int(val))

    # ── Summary table ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Summary Table")
    table_rows = []
    for variant_label, variant_key in [
        ("With categorical",    "withcat"),
        ("Without categorical", "withoutcat"),
    ]:
        row = {"Model": variant_label}
        for metal in METALS:
            val = results[metal][variant_key]
            if isinstance(val, float):
                row[f"{metal} (%)"] = f"{val:.2f}"
            elif val is None:
                row[f"{metal} (%)"] = "—"
            else:
                row[f"{metal} (%)"] = "Error"
        table_rows.append(row)

    df_summary = pd.DataFrame(table_rows).set_index("Model")
    st.dataframe(df_summary, use_container_width=True)

    with st.expander("📄 Raw input values"):
        st.dataframe(
            pd.DataFrame([row_withcat]).T.rename(columns={0: "Value"}),
            use_container_width=True,
        )
