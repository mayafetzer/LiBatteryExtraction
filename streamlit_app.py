import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Li Battery Extraction – Train & Predict",
    page_icon="🔋",
    layout="wide",
)

METALS       = ["Li", "Co", "Mn", "Ni"]
METAL_COLORS = {"Li": "#4e9af1", "Co": "#e05c5c", "Mn": "#7bc67e", "Ni": "#f0a500"}

# Expected column names (flexible matching done below)
CAT_COLS = ["Leaching agent", "Type of reducing agent"]
NUM_COLS = ["Li in feed %", "Co in feed %", "Mn in feed %", "Ni in feed %",
            "Concentration, M", "Concentration %", "Time,min", "Temperature, C"]
ALL_FEAT = NUM_COLS + CAT_COLS
TARGET_COLS = METALS   # ["Li", "Co", "Mn", "Ni"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def fuzzy_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort column name normalisation."""
    rename = {}
    for col in df.columns:
        c = col.strip()
        rename[col] = c
    df = df.rename(columns=rename)
    return df

def build_pipeline(cat_cols, num_cols, model):
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("pre", pre), ("model", model)])

def train_models(df, progress_bar):
    """Train withcat and nocat models for each metal. Returns dict of models + metrics."""
    results   = {}
    step      = 0
    total     = len(METALS) * 2

    for metal in METALS:
        results[metal] = {}
        if metal not in df.columns:
            for v in ["withcat", "nocat"]:
                results[metal][v] = {"model": None, "r2": None, "mae": None, "rmse": None}
            step += 2
            continue

        y = df[metal].dropna()
        idx = y.index

        for variant in ["withcat", "nocat"]:
            step += 1
            progress_bar.progress(step / total, text=f"Training {metal} ({variant})…")

            if variant == "withcat":
                feat_cols = [c for c in ALL_FEAT if c in df.columns]
                cat_use   = [c for c in CAT_COLS if c in feat_cols]
                num_use   = [c for c in NUM_COLS if c in feat_cols]
            else:
                feat_cols = [c for c in NUM_COLS if c in df.columns]
                cat_use   = []
                num_use   = feat_cols

            X = df.loc[idx, feat_cols]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            best_pipe, best_r2 = None, -np.inf
            for Model, params in [
                (GradientBoostingRegressor,
                 {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                  "subsample": 0.8, "random_state": 42}),
                (RandomForestRegressor,
                 {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 2,
                  "random_state": 42, "n_jobs": -1}),
            ]:
                pipe = build_pipeline(cat_use, num_use, Model(**params))
                cv   = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
                if cv.mean() > best_r2:
                    best_r2  = cv.mean()
                    best_pipe = pipe

            best_pipe.fit(X_train, y_train)
            y_pred = best_pipe.predict(X_test)
            r2   = r2_score(y_test, y_pred)
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results[metal][variant] = {
                "model":      best_pipe,
                "r2":         r2,
                "mae":        mae,
                "rmse":       rmse,
                "feat_cols":  feat_cols,
                "cat_use":    cat_use,
                "num_use":    num_use,
            }

    return results

# ─────────────────────────────────────────────────────────────────────────────
st.title("🔋 Li Battery Extraction – Train & Predict")
st.markdown(
    "Upload your leaching dataset, train models for **Li, Co, Mn, and Ni** "
    "extraction, then predict for new conditions."
)
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "trained" not in st.session_state:
    st.session_state.trained   = False
if "models"  not in st.session_state:
    st.session_state.models    = {}
if "cat_options" not in st.session_state:
    st.session_state.cat_options = {
        "Leaching agent":         ["H2SO4", "HCl", "HNO3", "H3PO4", "Citric acid",
                                   "Acetic acid", "Oxalic acid", "Tartaric acid",
                                   "DL-malic acid", "Ascorbic acid", "Succinic acid"],
        "Type of reducing agent": ["None", "H2O2", "Glucose", "Ascorbic acid",
                                   "Sucrose", "Starch", "Na2SO3", "Fe", "Al"],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Upload
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("① Upload dataset")
uploaded = st.file_uploader(
    "Excel file (.xlsx / .xls) — must contain feature columns and target columns (Li, Co, Mn, Ni)",
    type=["xlsx", "xls"],
)

if uploaded:
    try:
        df_raw = pd.read_excel(uploaded, header=0, skiprows=[1])
        df_raw = fuzzy_rename(df_raw)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.success(f"Loaded **{len(df_raw):,} rows × {len(df_raw.columns)} columns**")

    with st.expander("📄 Data preview", expanded=True):
        st.dataframe(df_raw.head(10), use_container_width=True)

    # Column detection
    found_feat    = [c for c in ALL_FEAT    if c in df_raw.columns]
    found_targets = [c for c in TARGET_COLS if c in df_raw.columns]
    missing_feat  = [c for c in ALL_FEAT    if c not in df_raw.columns]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**✅ Recognised columns**")
        st.write(", ".join(found_feat + found_targets) or "None")
    with col_b:
        if missing_feat:
            st.markdown("**⚠️ Not found (will be skipped)**")
            st.write(", ".join(missing_feat))

    if not found_targets:
        st.error("No target columns (Li / Co / Mn / Ni) found. Check column names.")
        st.stop()

    # Pull unique categorical values from data for the prediction dropdowns
    for col in CAT_COLS:
        if col in df_raw.columns:
            vals = sorted(df_raw[col].dropna().unique().tolist())
            st.session_state.cat_options[col] = vals

    # ── Train button ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("② Train models")
    if st.button("🚀 Train all models", type="primary", use_container_width=True):
        bar = st.progress(0, text="Starting…")
        with st.spinner("Training in progress…"):
            trained = train_models(df_raw, bar)
        bar.empty()
        st.session_state.models  = trained
        st.session_state.trained = True
        st.success("All models trained!")

    # ── Training metrics ──────────────────────────────────────────────────────
    if st.session_state.trained:
        st.subheader("📈 Model performance (held-out test set)")
        rows = []
        for metal in METALS:
            for variant in ["withcat", "nocat"]:
                info = st.session_state.models.get(metal, {}).get(variant, {})
                r2   = info.get("r2")
                rows.append({
                    "Metal":   metal,
                    "Variant": variant,
                    "R²":      f"{r2:.3f}"  if isinstance(r2, float) else "—",
                    "MAE (%)": f"{info.get('mae', 0):.2f}" if isinstance(info.get("mae"), float) else "—",
                    "RMSE (%)":f"{info.get('rmse',0):.2f}" if isinstance(info.get("rmse"),float) else "—",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Predict
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.trained:
    st.divider()
    st.subheader("③ Predict new conditions")

    with st.form("predict_form"):
        st.markdown("**Feed composition**")
        c1, c2, c3, c4 = st.columns(4)
        li_feed = c1.number_input("Li in feed (%)",  0.0, 100.0, 7.0,  0.1)
        co_feed = c2.number_input("Co in feed (%)",  0.0, 100.0, 20.0, 0.1)
        mn_feed = c3.number_input("Mn in feed (%)",  0.0, 100.0, 10.0, 0.1)
        ni_feed = c4.number_input("Ni in feed (%)",  0.0, 100.0, 10.0, 0.1)

        st.markdown("**Leaching conditions**")
        c5, c6, c7, c8 = st.columns(4)
        leaching_agent = c5.selectbox(
            "Leaching agent", st.session_state.cat_options["Leaching agent"]
        )
        leaching_conc  = c6.number_input("Concentration (M)",        0.0, 20.0,  2.0, 0.1)
        reducing_agent = c7.selectbox(
            "Reducing agent", st.session_state.cat_options["Type of reducing agent"]
        )
        reducing_conc  = c8.number_input("Reducing agent conc. (%)", 0.0, 100.0, 4.0, 0.5)

        c9, c10, _ = st.columns([1, 1, 2])
        time_min = c9.number_input("Time (min)",        0.0, 600.0, 60.0, 5.0)
        temp_c   = c10.number_input("Temperature (°C)", 0.0, 200.0, 80.0, 5.0)

        submitted = st.form_submit_button(
            "🔮 Predict extraction efficiencies", type="primary", use_container_width=True
        )

    if submitted:
        row_all = {
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

        st.divider()
        st.subheader("📊 Predicted Extraction Efficiencies")

        pred_cols = st.columns(len(METALS))
        summary   = []

        for i, metal in enumerate(METALS):
            with pred_cols[i]:
                color = METAL_COLORS[metal]
                st.markdown(
                    f"<h3 style='color:{color};text-align:center'>{metal}</h3>",
                    unsafe_allow_html=True,
                )
                for variant in ["withcat", "nocat"]:
                    info  = st.session_state.models.get(metal, {}).get(variant, {})
                    model = info.get("model")
                    label = "With categorical" if variant == "withcat" else "No categorical"

                    if model is None:
                        st.markdown(f"**{label}:** *(not available)*")
                        summary.append({"Metal": metal, "Variant": label, "Predicted (%)": "—"})
                        continue

                    feat_cols = info["feat_cols"]
                    X_pred    = pd.DataFrame([{k: row_all[k] for k in feat_cols if k in row_all}])
                    try:
                        val = float(np.clip(model.predict(X_pred)[0], 0, 100))
                        st.metric(label, f"{val:.1f}%")
                        st.progress(int(val))
                        summary.append({"Metal": metal, "Variant": label, "Predicted (%)": f"{val:.2f}"})
                    except Exception as e:
                        st.markdown(f"**{label}:** ⚠️ {e}")
                        summary.append({"Metal": metal, "Variant": label, "Predicted (%)": "Error"})

        st.divider()
        st.subheader("📋 Summary")
        df_sum = pd.DataFrame(summary).pivot(index="Variant", columns="Metal", values="Predicted (%)")
        st.dataframe(df_sum, use_container_width=True)
