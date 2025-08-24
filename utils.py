
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from math import radians, cos, sin, asin, sqrt

RANDOM_STATE = 42

# -------------------- Preprocessing --------------------
def prepare_data(smart_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Merge smart logs with metadata and ensure dtypes."""
    df = smart_df.copy()
    # Parse date
    df["date"] = pd.to_datetime(df["date"])
    # Ensure expected columns exist
    required = ["tower_id","zone","season","day_of_week","visit_order","prev_tower","weather",
                "precip_mm","wind_kmph","temp_c","load_level","load_category","any_issue","x","y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in smart_tower_dataset: {missing}")
    # Join metadata (unique per tower)
    meta_cols = [c for c in meta_df.columns if c not in ["zone"]]  # avoid duplicate zone naming
    df = df.merge(meta_df[meta_cols + ["tower_id"]], on="tower_id", how="left", suffixes=("","_meta"))
    # Coerce target to int
    if df["any_issue"].dtype != int and df["any_issue"].dtype != bool:
        df["any_issue"] = (df["any_issue"].astype(str).str.strip().isin(["1","True","true","Y","y","yes"])).astype(int)
    else:
        df["any_issue"] = df["any_issue"].astype(int)
    return df

def train_val_split(df: pd.DataFrame, test_size=0.2):
    """Time-aware split by date to reduce leakage."""
    df = df.sort_values("date")
    cutoff = int(len(df) * (1 - test_size))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()

# -------------------- Feature pipeline (tabular baseline) --------------------
def make_tabular_features():
    num_cols = ["precip_mm","wind_kmph","temp_c","load_level","voltage_v","frequency_hz",
                "x","y","elevation_m","age_years","maintenance_score","tree_density","wind_exposure",
                "flood_risk","corrosion_risk","urban_density","distance_to_substation_km"]
    num_cols = [c for c in num_cols if c]  # filter None
    cat_cols = ["season","day_of_week","zone","weather","load_category"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), [c for c in num_cols if c in ["precip_mm","wind_kmph","temp_c","load_level","voltage_v","frequency_hz","x","y",
                                                                "elevation_m","age_years","maintenance_score","tree_density","wind_exposure",
                                                                "flood_risk","corrosion_risk","urban_density","distance_to_substation_km"]]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in cat_cols if c in ["season","day_of_week","zone","weather","load_category"]]),
    ])
    return pre

# -------------------- Sequences for LSTM --------------------
def build_daily_sequences(df: pd.DataFrame, by=["zone","date"]):
    """Return list of sequences; each is DataFrame sorted by visit_order within a (zone,date) group."""
    seqs = []
    for _, g in df.sort_values(["visit_order"]).groupby(by):
        seqs.append(g.sort_values("visit_order"))
    return seqs

def pack_lstm_tensors(seqs, feature_cols, target_col="any_issue"):
    """Convert per-day DataFrames into padded numpy arrays: X:[B,T,F], y:[B,T]."""
    X_list, y_list, tower_list = [], [], []
    max_len = max(len(s) for s in seqs)
    for s in seqs:
        X = s[feature_cols].to_numpy(dtype=np.float32)
        y = s[target_col].to_numpy(dtype=np.float32)
        # pad
        pad_len = max_len - len(s)
        if pad_len > 0:
            X = np.pad(X, ((0,pad_len),(0,0)), mode="constant")
            y = np.pad(y, (0,pad_len), mode="constant")
        X_list.append(X)
        y_list.append(y)
        tower_list.append(s["tower_id"].tolist() + [None]*(pad_len))
    X = np.stack(X_list)  # [B,T,F]
    y = np.stack(y_list)  # [B,T]
    return X, y, max_len, tower_list

# -------------------- Markov Chain --------------------
def fit_markov(df: pd.DataFrame, by=["zone","season","weather"]):
    """Return dictionary of transition probabilities P(next | current)."""
    trans = {}
    grouped = df.sort_values("visit_order").groupby(by)
    for key, g in grouped:
        # transitions from prev_tower -> tower_id
        pairs = g[["prev_tower","tower_id"]].dropna()
        counts = pairs.value_counts().reset_index(name="cnt")
        d = {}
        for curr, sub in counts.groupby("prev_tower"):
            tot = sub["cnt"].sum()
            d[int(curr)] = {int(r.tower_id): r.cnt / tot for _, r in sub.iterrows()}
        trans[key] = d
    return trans

def predict_next_from_markov(trans, context_key, current_tower):
    # backoff: ignore weather, then season
    keys = [context_key]
    if len(context_key) == 3:
        keys.append((context_key[0], context_key[1], None))
        keys.append((context_key[0], None, None))
    for k in keys:
        if k in trans and current_tower in trans[k]:
            probs = trans[k][current_tower]
            # return tower with max prob
            return max(probs.items(), key=lambda x: x[1])
    return None

# -------------------- Distance & Route Optimization --------------------
def pairwise_distance_matrix(towers_df: pd.DataFrame, x_col="x", y_col="y"):
    """Euclidean distance matrix based on planar coordinates (x,y)."""
    coords = towers_df[[x_col, y_col]].to_numpy(dtype=float)
    n = len(coords)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            D[i, j] = D[j, i] = d
    return D

def nearest_neighbor_route(D: np.ndarray, start_idx=0):
    n = D.shape[0]
    unvisited = set(range(n))
    route = [start_idx]
    unvisited.remove(start_idx)
    while unvisited:
        last = route[-1]
        next_idx = min(unvisited, key=lambda j: D[last, j])
        route.append(next_idx)
        unvisited.remove(next_idx)
    return route

def two_opt(route, D):
    improved = True
    best = route[:]
    best_len = route_length(best, D)
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for k in range(i+1, len(best) - 1):
                new_route = best[:i] + best[i:k+1][::-1] + best[k+1:]
                new_len = route_length(new_route, D)
                if new_len < best_len:
                    best, best_len = new_route, new_len
                    improved = True
                    break
            if improved:
                break
    return best

def route_length(route, D):
    return sum(D[route[i], route[i+1]] for i in range(len(route)-1))

def optimize_route(towers_df: pd.DataFrame, start_from_highest=True):
    """Return ordered tower_ids for visit minimizing distance (NN + 2-opt)."""
    towers_df = towers_df.reset_index(drop=True)
    D = pairwise_distance_matrix(towers_df)
    start_idx = 0
    if start_from_highest and "risk_score" in towers_df.columns:
        start_idx = towers_df["risk_score"].idxmax()
    nn = nearest_neighbor_route(D, start_idx=start_idx)
    opt = two_opt(nn, D)
    return towers_df.iloc[opt]["tower_id"].tolist(), D

# -------------------- Utility --------------------
def top_k_by_threshold(pred_df: pd.DataFrame, k=None, threshold=0.5):
    """Pick towers above threshold, if none then top-k by score."""
    above = pred_df[pred_df["risk_score"] >= threshold]
    if len(above) > 0:
        return above.sort_values("risk_score", ascending=False)
    if k is None:
        k = min(10, len(pred_df))
    return pred_df.nlargest(k, "risk_score")

