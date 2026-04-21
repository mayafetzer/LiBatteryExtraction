import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Li Battery Extraction – Train & Predict",
    page_icon="🔋",
    layout="wide",
)

METALS = ["Li", "Co", "Mn", "Ni"]

CAT_COLS = ["Leaching agent", "Type of reducing agent"]

NUM_COLS = [
    "Li in feed %",
    "Co in feed %",
    "Mn in feed %",
    "Ni in feed %",
    "Concentration, M",
    "Concentration %",
    "Time,min",
    "Temperature, C",
]

ALL_FEAT = NUM_COLS + CAT_COLS

# ─────────────────────────────────────────────────────────────
def clean_columns(df):
    """Normalize column names: strip whitespace and collapse internal spaces."""
    import re
    df.columns = [re.sub(r'\s+', ' ', c).strip() for c in df.columns]
    return df


def clean_data(df):
    """Coerce numeric columns, clean categorical columns."""
    # Force numeric conversion on columns that may have come in as object
    for col in NUM_COLS + METALS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strip whitespace from categoricals; replace whitespace-only strings with NaN
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    return df

# ─────────────────────────────────────────────────────────────
def build_pipeline(cat_cols, num_cols, model):
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    preprocessor = ColumnTransformer(transformers)
    return Pipeline([("pre", preprocessor), ("model", model)])

# ─────────────────────────────────────────────────────────────
def train_models(df):
    results = {}

    st.subheader("📊 Data availability")
    availability = {m: int(df[m].notna().sum()) for m in METALS if m in df.columns}
    st.write(availability)

    for metal in METALS:
        results[metal] = {}

        if metal not in df.columns:
            st.warning(f"{metal} column not found")
            for v in ["withcat", "nocat"]:
                results[metal][v] = {"model": None}
            continue

        for variant in ["withcat", "nocat"]:
            if variant == "withcat":
                feat_cols = [c for c in ALL_FEAT if c in df.columns]
                cat_cols  = [c for c in CAT_COLS if c in feat_cols]
                num_cols  = [c for c in NUM_COLS  if c in feat_cols]
            else:
                feat_cols = [c for c in NUM_COLS if c in df.columns]
                cat_cols  = []
                num_cols  = feat_cols

            if not feat_cols:
                st.warning(f"{metal} ({variant}) → no features")
                results[metal][variant] = {"model": None}
                continue

            subset = df[feat_cols + [metal]].copy()

            # Fill numeric NaNs with column median
            for col in num_cols:
                if col in subset.columns:
                    subset[col] = subset[col].fillna(subset[col].median())

            subset = subset.dropna()

            if len(subset) < 8:
                st.warning(f"{metal} ({variant}) skipped → only {len(subset)} rows after dropping NaN")
                results[metal][variant] = {"model": None}
                continue

            X = subset[feat_cols]
            y = subset[metal]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            n_folds = max(2, min(5, len(X_train) // 2))

            best_model = None
            best_score = -np.inf

            for Model, params in [
                (GradientBoostingRegressor, {"n_estimators": 150}),
                (RandomForestRegressor,     {"n_estimators": 150}),
            ]:
                pipe = build_pipeline(cat_cols, num_cols, Model(**params))
                try:
                    score = cross_val_score(pipe, X_train, y_train, cv=n_folds).mean()
                except Exception:
                    score = -np.inf

                if score > best_score:
                    best_score = score
                    best_model = pipe

            if best_model is None:
                best_model = build_pipeline(cat_cols, num_cols, RandomForestRegressor())

            best_model.fit(X_train, y_train)
            preds = best_model.predict(X_test)

            results[metal][variant] = {
                "model":     best_model,
                "feat_cols": feat_cols,
                "r2":        r2_score(y_test, preds),
                "mae":       mean_absolute_error(y_test, preds),
                "rmse":      np.sqrt(mean_squared_error(y_test, preds)),
            }

    return results

# ─────────────────────────────────────────────────────────────
st.title("🔋 Li Battery Extraction – Train & Predict")

uploaded = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded:
    try:
        # Row 0 is a merged category header; row 1 contains the real column names
        df = pd.read_excel(uploaded, header=1)
        df = clean_columns(df)
        df = clean_data(df)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())
    st.divider()

    if st.button("🚀 Train Models"):
        with st.spinner("Training…"):
            models = train_models(df)
        st.session_state.models  = models
        st.session_state.trained = True
        st.success("Training complete ✅")

# ─────────────────────────────────────────────────────────────
# 📈 RESULTS
if st.session_state.get("trained"):

    st.subheader("📈 Model Performance")

    rows = []
    for metal in METALS:
        for variant in ["withcat", "nocat"]:
            info = st.session_state.models.get(metal, {}).get(variant, {})
            rows.append({
                "Metal":   metal,
                "Variant": variant,
                "R²":   round(info["r2"],   3) if info.get("r2")   is not None else "—",
                "MAE":  round(info["mae"],  2) if info.get("mae")  is not None else "—",
                "RMSE": round(info["rmse"], 2) if info.get("rmse") is not None else "—",
            })
    st.dataframe(pd.DataFrame(rows))

    # ─────────────────────────────────────────────────────────
    # 🔮 PREDICTION
    st.subheader("🔮 Predict Extraction (%)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Material properties**")
        li_feed  = st.number_input("Li in feed %",  min_value=0.0, max_value=100.0, value=6.79)
        co_feed  = st.number_input("Co in feed %",  min_value=0.0, max_value=100.0, value=17.68)
        mn_feed  = st.number_input("Mn in feed %",  min_value=0.0, max_value=100.0, value=16.46)
        ni_feed  = st.number_input("Ni in feed %",  min_value=0.0, max_value=100.0, value=17.58)

    with col2:
        st.markdown("**Reaction conditions**")
        leach_agent  = st.selectbox("Leaching agent",       ["ORGANIC_ACID", "INORGANIC_ACID", "BASE"])
        reducing     = st.selectbox("Type of reducing agent", ["YES", "NO"])
        conc_m       = st.number_input("Concentration, M",  min_value=0.0, value=1.0)
        conc_pct     = st.number_input("Concentration %",   min_value=0.0, value=1.0)
        time_min     = st.number_input("Time, min",         min_value=0.0, value=60.0)
        temp_c       = st.number_input("Temperature, °C",   min_value=0.0, value=80.0)

    if st.button("🔮 Predict"):
        inputs = {
            "Li in feed %":         li_feed,
            "Co in feed %":         co_feed,
            "Mn in feed %":         mn_feed,
            "Ni in feed %":         ni_feed,
            "Leaching agent":       leach_agent,
            "Type of reducing agent": reducing,
            "Concentration, M":     conc_m,
            "Concentration %":      conc_pct,
            "Time,min":             time_min,
            "Temperature, C":       temp_c,
        }

        pred_rows = []
        for metal in METALS:
            for variant in ["withcat", "nocat"]:
                info  = st.session_state.models.get(metal, {}).get(variant, {})
                model = info.get("model")
                if model is None:
                    continue
                feat_cols = info["feat_cols"]
                X_pred = pd.DataFrame([{k: inputs[k] for k in feat_cols}])
                try:
                    pred = float(np.clip(model.predict(X_pred)[0], 0, 100))
                except Exception as e:
                    pred = None
                pred_rows.append({
                    "Metal":             metal,
                    "Variant":           variant,
                    "Predicted yield (%)": f"{pred:.1f}" if pred is not None else "—",
                })

        st.dataframe(pd.DataFrame(pred_rows))
