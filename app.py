
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from utils import prepare_data, make_tabular_features, build_daily_sequences, pack_lstm_tensors, fit_markov, optimize_route, top_k_by_threshold
from lstm_model import train_lstm, infer_lstm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import os

st.set_page_config(page_title="Predictive Route Planner", layout="wide")

st.title("🔮 Predictive Route Planner for Tower Inspections")
st.caption("Hybrid risk scoring (Markov + LSTM) + Route optimization (NN + 2-opt).")

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Risk model", ["Hybrid (LSTM + Markov)", "LSTM only", "Markov only", "LogReg baseline"])
    zone_filter = st.selectbox("Zone", ["All","North","South","East","West"])
    threshold = st.slider("Risk threshold", 0.0, 1.0, 0.5, 0.01)
    top_k = st.number_input("Fallback Top-K towers", min_value=5, max_value=50, value=10, step=1)
    epochs = st.slider("LSTM epochs (if training)", 1, 25, 8)
    st.divider()
    st.write("**Data**")
    data_src = st.radio("Data source", ["Use sample data", "Upload CSVs"])

def load_data():
    if data_src == "Use sample data":
        smart = pd.read_csv("sample_data/smart_tower_dataset.csv")
        meta = pd.read_csv("sample_data/towers_metadata.csv")
    else:
        smart = st.file_uploader("smart_tower_dataset.csv", type=["csv"], key="smart")
        meta = st.file_uploader("towers_metadata.csv", type=["csv"], key="meta")
        if smart and meta:
            smart = pd.read_csv(smart)
            meta = pd.read_csv(meta)
        else:
            st.stop()
    return smart, meta

smart_df, meta_df = load_data()

# Filters
if zone_filter != "All":
    smart_df = smart_df[smart_df["zone"] == zone_filter]

# Prepare
df = prepare_data(smart_df, meta_df)

# Today's context (user can override)
st.subheader("📆 Today's Context")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    today_date = st.date_input("Date", value=date.today())
with c2:
    season = st.selectbox("Season", sorted(df["season"].dropna().unique()))
with c3:
    dow = today_date.strftime("%A")
    st.text_input("Day of week", value=dow, disabled=True)
with c4:
    weather = st.selectbox("Weather", sorted(df["weather"].dropna().unique()))
with c5:
    load_cat = st.selectbox("Load Category", sorted(df["load_category"].dropna().unique()))

# --- Train models (cached via session state simple flag) ---
@st.cache_data(show_spinner=False)
def compute_markov(df_):
    return fit_markov(df_)

@st.cache_resource(show_spinner=False)
def train_baseline(df_):
    feature_cols = ["precip_mm","wind_kmph","temp_c","load_level","voltage_v","frequency_hz",
                    "x","y","elevation_m","age_years","maintenance_score","tree_density","wind_exposure",
                    "flood_risk","corrosion_risk","urban_density","distance_to_substation_km"]
    feature_cols = [c for c in feature_cols if c in df_.columns]
    cat_cols = ["season","day_of_week","zone","weather","load_category"]
    X_cols = feature_cols + cat_cols
    df_train, df_val = df_.pipe(lambda d: d.sort_values("date")).pipe(lambda d: (d.iloc[:int(0.8*len(d))], d.iloc[int(0.8*len(d)):]))
    pre = make_tabular_features()
    clf = LogisticRegression(max_iter=200, n_jobs=None)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(df_train[X_cols], df_train["any_issue"])
    val_pred = pipe.predict_proba(df_val[X_cols])[:,1]
    try:
        auc = roc_auc_score(df_val["any_issue"], val_pred)
    except:
        auc = float("nan")
    return pipe, X_cols, auc

@st.cache_resource(show_spinner=False)
def train_lstm_model(df_):
    # Build sequences
    feature_cols = ["precip_mm","wind_kmph","temp_c","load_level","voltage_v","frequency_hz",
                    "elevation_m","age_years","maintenance_score","tree_density","wind_exposure",
                    "flood_risk","corrosion_risk","urban_density","distance_to_substation_km"]
    feature_cols = [c for c in feature_cols if c in df_.columns]
    # Normalize numerical features
    Xn = df_[feature_cols]
    Xn = (Xn - Xn.mean()) / (Xn.std() + 1e-6)
    dfn = df_.copy()
    dfn[feature_cols] = Xn
    seqs = build_daily_sequences(dfn, by=["zone","date"])
    X, y, T, towers = pack_lstm_tensors(seqs, feature_cols, target_col="any_issue")
    # time-aware split
    n = len(seqs)
    n_train = int(0.8*n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    model = train_lstm(X_train, y_train, X_val, y_val, epochs=epochs)
    return model, feature_cols

markov = compute_markov(df)

if model_choice in ["Hybrid (LSTM + Markov)", "LSTM only"]:
    with st.spinner("Training LSTM (cached)..."):
        lstm_model, lstm_feats = train_lstm_model(df)
else:
    lstm_model, lstm_feats = None, []

if model_choice in ["Hybrid (LSTM + Markov)", "LogReg baseline"]:
    with st.spinner("Training baseline logistic model (cached)..."):
        logreg_pipe, X_cols, auc = train_baseline(df)
        st.caption(f"Baseline validation AUC: {auc:.3f}")
else:
    logreg_pipe, X_cols = None, []

# ---- Predict today's risk for each tower ----
st.subheader("🧠 Predicted Risk by Tower")
# Construct a "today" frame per tower from metadata + today's context
towers = meta_df[["tower_id","x","y"]].copy()
towers["zone"] = meta_df["zone"]
towers["season"] = season
towers["day_of_week"] = dow
towers["weather"] = weather
# approximate load_level from metadata or set median
median_load = float(df["load_level"].median()) if "load_level" in df.columns else 0.0
towers["load_level"] = median_load
towers["load_category"] = load_cat
# Join remaining meta columns
for col in ["elevation_m","age_years","maintenance_score","tree_density","wind_exposure","flood_risk","corrosion_risk","urban_density","distance_to_substation_km"]:
    if col in meta_df.columns:
        towers[col] = meta_df[col]

# Risk via LSTM: use last known sequence context (approx: use historical mean features per tower)
# 🔮 LSTM risk scoring
# 🔮 LSTM risk scoring
if lstm_model is not None and lstm_feats:
    # Create a DataFrame with all training features
    Xl = pd.DataFrame(index=towers.index)

    for f in lstm_feats:
        if f in towers.columns:
            Xl[f] = towers[f]
        else:
            # feature missing at inference → fill with 0
            Xl[f] = 0.0

    # Reorder to match training feature order
    Xl = Xl[lstm_feats]

    # Normalize with training stats (from df)
    mu = df[lstm_feats].mean()
    sigma = df[lstm_feats].std() + 1e-6
    Xl = (Xl - mu) / sigma

    # [B,T,F] format
    X_seq = Xl.to_numpy(dtype=np.float32)[None, ...]
    lstm_probs = infer_lstm(lstm_model, X_seq).ravel()
    towers["risk_lstm"] = lstm_probs



# Risk via LogReg baseline
if logreg_pipe is not None:
    Xt = towers[[c for c in X_cols if c in towers.columns]]
    base_probs = logreg_pipe.predict_proba(Xt)[:,1]
else:
    base_probs = np.zeros(len(towers))

# Risk via Markov next-tower heuristic: boost towers that frequently follow recently visited towers
# Find most recent day for the selected zone and build boost scores
df_zone = df if zone_filter=="All" else df[df["zone"]==zone_filter]
recent_day = df_zone["date"].max()
df_recent = df_zone[df_zone["date"]==recent_day].sort_values("visit_order")
boost = np.zeros(len(towers), dtype=float)
if len(df_recent) > 0:
    for _, row in df_recent.iterrows():
        key = (row["zone"], row["season"], row["weather"])
        curr = int(row["tower_id"])
        pred = None
        # locate probabilities for all possible next
        if key in markov and curr in markov[key]:
            for tid, p in markov[key][curr].items():
                idx = towers.index[towers["tower_id"]==tid]
                if len(idx)>0:
                    boost[idx[0]] += p * 0.2  # small boost

# Combine according to selection
if model_choice == "Hybrid (LSTM + Markov)":
    risk = 0.6*lstm_probs + 0.4*base_probs + boost
elif model_choice == "LSTM only":
    risk = lstm_probs
elif model_choice == "Markov only":
    risk = boost  # purely transitions
else:
    risk = base_probs

pred_df = towers.copy()
pred_df["risk_score"] = np.clip(risk, 0, 1)
pred_df = pred_df.sort_values("risk_score", ascending=False)

st.dataframe(pred_df[["tower_id","zone","risk_score"]].head(25), use_container_width=True)

# ---- Route Optimization on selected towers ----
st.subheader("🗺️ Suggested Route")
selected = top_k_by_threshold(pred_df, k=int(top_k), threshold=float(threshold)).copy()
route, D = optimize_route(selected, start_from_highest=True)
st.write("**Visit order:**", " → ".join(map(str, route)))

# Map (pydeck) if x,y look like lat/lon (range approx), else scatter
minx, maxx = float(pred_df["x"].min()), float(pred_df["x"].max())
miny, maxy = float(pred_df["y"].min()), float(pred_df["y"].max())
is_latlon = (-90 <= miny <= 90) and (-90 <= maxy <= 90) and (-180 <= minx <= 180) and (-180 <= maxx <= 180)
if is_latlon:
    import pydeck as pdk
    mdf = selected.copy()
    mdf["order"] = 0
    order_map = {tid:i for i, tid in enumerate(route)}
    mdf["order"] = mdf["tower_id"].map(order_map)
    mdf = mdf.sort_values("order")
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=mdf["y"].mean(), longitude=mdf["x"].mean(), zoom=8, pitch=0),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=mdf,
                get_position='[x, y]',
                get_radius=500,
            ),
            pdk.Layer(
                "PathLayer",
                data=[{"path": mdf[["x","y"]].to_dict(orient="records")}],
                get_width=5,
            ),
        ],
    ))
else:
    st.info("Coordinates appear to be planar (x,y). Map view disabled; showing DataFrame instead.")
    st.dataframe(selected[["tower_id","x","y","risk_score"]])

st.success("Route ready. You can export predictions below.")

# ---- Export ----
csv = pred_df.to_csv(index=False).encode("utf-8")
st.download_button("Download full risk table (CSV)", data=csv, file_name="predicted_risks.csv", mime="text/csv")

st.caption("Tip: Commit this project to GitHub and deploy on Streamlit Community Cloud.")

