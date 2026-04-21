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
# 🔧 Safe column cleanup (minimal, not destructive)
def clean_columns(df):
    df.columns = [c.strip() for c in df.columns]
    return df

# ─────────────────────────────────────────────────────────────
def build_pipeline(cat_cols, num_cols, model):

    transformers = []

    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))

    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    preprocessor = ColumnTransformer(transformers)

    return Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])

# ─────────────────────────────────────────────────────────────
def train_models(df):

    results = {}

    st.subheader("📊 Data availability")

    availability = {}
    for m in METALS:
        if m in df.columns:
            availability[m] = int(df[m].notna().sum())
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
                cat_cols = [c for c in CAT_COLS if c in feat_cols]
                num_cols = [c for c in NUM_COLS if c in feat_cols]
            else:
                feat_cols = [c for c in NUM_COLS if c in df.columns]
                cat_cols = []
                num_cols = feat_cols

            if not feat_cols:
                st.warning(f"{metal} ({variant}) → no features")
                results[metal][variant] = {"model": None}
                continue

            subset = df[feat_cols + [metal]].copy()

            # ✅ Fill numeric NaNs safely
            for col in num_cols:
                if col in subset.columns:
                    subset[col] = subset[col].fillna(subset[col].median())

            subset = subset.dropna()

            if len(subset) < 8:
                st.warning(f"{metal} ({variant}) skipped → only {len(subset)} rows")
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
                (RandomForestRegressor, {"n_estimators": 150})
            ]:

                pipe = build_pipeline(cat_cols, num_cols, Model(**params))

                try:
                    scores = cross_val_score(pipe, X_train, y_train, cv=n_folds)
                    score = scores.mean()
                except Exception as e:
                    st.warning(f"{metal} CV failed → fallback used")
                    score = -np.inf

                if score > best_score:
                    best_score = score
                    best_model = pipe

            # ✅ fallback model
            if best_model is None:
                best_model = build_pipeline(cat_cols, num_cols, RandomForestRegressor())

            best_model.fit(X_train, y_train)

            preds = best_model.predict(X_test)

            results[metal][variant] = {
                "model": best_model,
                "r2": r2_score(y_test, preds),
                "mae": mean_absolute_error(y_test, preds),
                "rmse": np.sqrt(mean_squared_error(y_test, preds)),
                "feat_cols": feat_cols
            }

    return results

# ─────────────────────────────────────────────────────────────
st.title("🔋 Li Battery Extraction – Train & Predict")

uploaded = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded:

    try:
        df = pd.read_excel(uploaded, header=0)
        df = clean_columns(df)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success(f"Loaded {df.shape[0]} rows")

    st.dataframe(df.head())

    st.divider()

    if st.button("🚀 Train Models"):

        with st.spinner("Training..."):
            models = train_models(df)

        st.session_state.models = models
        st.session_state.trained = True
        st.success("Training complete")

# ─────────────────────────────────────────────────────────────
# 📈 RESULTS
if st.session_state.get("trained"):

    st.subheader("📈 Model Performance")

    rows = []

    for metal in METALS:
        for variant in ["withcat", "nocat"]:

            info = st.session_state.models.get(metal, {}).get(variant, {})

            rows.append({
                "Metal": metal,
                "Variant": variant,
                "R²": round(info.get("r2", np.nan), 3) if info.get("r2") else "—",
                "MAE": round(info.get("mae", np.nan), 2) if info.get("mae") else "—",
                "RMSE": round(info.get("rmse", np.nan), 2) if info.get("rmse") else "—",
            })

    st.dataframe(pd.DataFrame(rows))

    # ─────────────────────────────────────────────────────────
    # 🔮 PREDICTION
    st.subheader("🔮 Predict")

    with st.form("predict"):

        inputs = {}

        for col in NUM_COLS:
            inputs[col] = st.number_input(col, value=0.0)

        for col in CAT_COLS:
            inputs[col] = st.text_input(col, value="")

        submitted = st.form_submit_button("Predict")

    if submitted:

        results = []

        for metal in METALS:
            for variant in ["withcat", "nocat"]:

                info = st.session_state.models.get(metal, {}).get(variant, {})
                model = info.get("model")

                if model is None:
                    continue

                feat_cols = info["feat_cols"]

                X_pred = pd.DataFrame([{k: inputs[k] for k in feat_cols}])

                try:
                    pred = float(np.clip(model.predict(X_pred)[0], 0, 100))
                except:
                    pred = None

                results.append({
                    "Metal": metal,
                    "Variant": variant,
                    "Prediction (%)": pred
                })

        st.dataframe(pd.DataFrame(results))
