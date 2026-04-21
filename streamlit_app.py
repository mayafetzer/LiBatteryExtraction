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
    "Li in feed %", "Co in feed %", "Mn in feed %", "Ni in feed %",
    "Concentration, M", "Concentration %", "Time,min", "Temperature, C"
]

# ─────────────────────────────────────────────────────────────
# 🔧 Improved column cleaner
def clean_columns(df):
    mapping = {}
    for col in df.columns:
        c = col.strip().lower()

        if "li" in c and "feed" not in c:
            mapping[col] = "Li"
        elif "co" in c and "feed" not in c:
            mapping[col] = "Co"
        elif "mn" in c and "feed" not in c:
            mapping[col] = "Mn"
        elif "ni" in c and "feed" not in c:
            mapping[col] = "Ni"
        else:
            mapping[col] = col.strip()

    return df.rename(columns=mapping)

# ─────────────────────────────────────────────────────────────
def build_pipeline(cat_cols, num_cols, model):
    transformers = []

    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))

    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    pre = ColumnTransformer(transformers)

    return Pipeline([
        ("pre", pre),
        ("model", model)
    ])

# ─────────────────────────────────────────────────────────────
def train_models(df):

    results = {}

    st.write("### 📊 Data availability per metal")
    for metal in METALS:
        if metal in df.columns:
            st.write(f"{metal}: {df[metal].notna().sum()} valid rows")

    for metal in METALS:
        results[metal] = {}

        if metal not in df.columns:
            st.error(f"{metal} column NOT FOUND")
            continue

        for variant in ["withcat", "nocat"]:

            if variant == "withcat":
                feat_cols = [c for c in NUM_COLS + CAT_COLS if c in df.columns]
                cat_cols = [c for c in CAT_COLS if c in feat_cols]
                num_cols = [c for c in NUM_COLS if c in feat_cols]
            else:
                feat_cols = [c for c in NUM_COLS if c in df.columns]
                cat_cols = []
                num_cols = feat_cols

            if not feat_cols:
                st.warning(f"{metal} ({variant}) → no usable features")
                continue

            subset = df[feat_cols + [metal]].copy()

            # Fill numeric NaNs with median
            subset[num_cols] = subset[num_cols].fillna(subset[num_cols].median())

            # Drop remaining NaNs
            subset = subset.dropna()

            if len(subset) < 10:
                st.error(f"{metal} ({variant}) skipped: only {len(subset)} rows")
                continue

            X = subset[feat_cols]
            y = subset[metal]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            n_folds = max(2, min(5, len(X_train) // 3))

            best_model = None
            best_score = -np.inf

            for Model, params in [
                (GradientBoostingRegressor, {"n_estimators": 200}),
                (RandomForestRegressor, {"n_estimators": 200})
            ]:

                pipe = build_pipeline(cat_cols, num_cols, Model(**params))

                try:
                    scores = cross_val_score(pipe, X_train, y_train, cv=n_folds)
                    score = scores.mean()
                except Exception as e:
                    st.warning(f"{metal} CV failed → {e}")
                    score = -np.inf

                if score > best_score:
                    best_score = score
                    best_model = pipe

            if best_model is None:
                st.error(f"{metal} failed completely")
                continue

            best_model.fit(X_train, y_train)

            preds = best_model.predict(X_test)

            results[metal][variant] = {
                "model": best_model,
                "r2": r2_score(y_test, preds),
                "mae": mean_absolute_error(y_test, preds),
                "rmse": np.sqrt(mean_squared_error(y_test, preds)),
                "features": feat_cols
            }

    return results

# ─────────────────────────────────────────────────────────────
st.title("🔋 Li Battery Extraction – Train & Predict")

uploaded = st.file_uploader("Upload Excel", type=["xlsx", "xls"])

if uploaded:
    df = pd.read_excel(upload)
    df = clean_columns(df)

    st.success("File loaded")
    st.dataframe(df.head())

    if st.button("Train models"):

        models = train_models(df)
        st.session_state.models = models
        st.session_state.trained = True

# ─────────────────────────────────────────────────────────────
if st.session_state.get("trained"):

    st.subheader("📈 Results")

    rows = []

    for metal in METALS:
        for variant in ["withcat", "nocat"]:
            info = st.session_state.models.get(metal, {}).get(variant, {})

            rows.append({
                "Metal": metal,
                "Variant": variant,
                "R2": info.get("r2"),
                "MAE": info.get("mae"),
                "RMSE": info.get("rmse")
            })

    st.dataframe(pd.DataFrame(rows))
