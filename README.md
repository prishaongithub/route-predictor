
# Predictive Route Planner (Hybrid LSTM + Markov)

A Streamlit app that scores towers by outage/issue risk using a hybrid of:
- **LSTM** over daily visit sequences (learns sequential patterns)
- **Markov** transition boosts based on most recent routes
- **Logistic Regression** baseline using tabular features

Then it proposes an **optimized route** over the top-risk towers using Nearest Neighbor + 2-opt (TSP heuristic).

## Project Structure

```
.
├── app.py
├── lstm_model.py
├── utils.py
├── models/               # (optional) saved models
├── sample_data/
│   ├── smart_tower_dataset.csv
│   └── towers_metadata.csv
└── requirements.txt
```

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy via GitHub → Streamlit Cloud

1. Create a new **public** GitHub repo and add these files.
2. In the repo root, ensure `requirements.txt` and `app.py` exist.
3. Go to **https://streamlit.io/cloud** → “New app” → connect your GitHub repo → pick `app.py`.
4. Hit **Deploy**.

## Data Notes
- Place your CSVs in `sample_data/` or upload via the sidebar.
- Required columns in `smart_tower_dataset.csv`:
  `date, season, day_of_week, zone, visit_order, prev_tower, tower_id, x, y, weather, precip_mm, wind_kmph, temp_c, load_level, load_category, voltage_v, frequency_hz, any_issue`
- Required columns in `towers_metadata.csv`:
  `tower_id, zone, x, y` (plus optional: `elevation_m, age_years, maintenance_score, tree_density, wind_exposure, flood_risk, corrosion_risk, urban_density, distance_to_substation_km`).

## How it Works (Short)
- We build per-day sequences `(zone, date)` sorted by `visit_order` for the LSTM.
- The logistic baseline encodes numerics (scaled) + categoricals (one-hot).
- The Markov model learns `P(next | current, zone, season, weather)` and provides a small **boost** to towers likely to be next given the most recent route context.
- Route optimization selects towers above a risk **threshold** (fallback to Top-K) and orders them using **Nearest Neighbor + 2-opt** over the coordinate distances.

## Customize
- Tweak risk blending in `app.py` (weights 0.6/0.4 for LSTM/LogReg + Markov boost).
- Replace the optimizer with OR-Tools if you prefer an exact solver.
- Add caching of trained models to `models/` for faster startups.
