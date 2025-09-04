# scripts/03_predict_matchup.py
from pathlib import Path
import pandas as pd
import numpy as np
import joblib, json

DATA_PATH = Path("data/processed/team_games_2022_2024.parquet")
MODEL_PATH = Path("models/final_model.pkl")
FEATURES_PATH = Path("models/features.json")
OUTPUT_DIR = Path("predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_artifacts():
    df = pd.read_parquet(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    features = json.load(open(FEATURES_PATH))
    return df, model, features

def last_n_rows_feature_matrix(df, team, feature_names, n=12, end_season=2024, override_is_home=None):
    """Return an aligned numeric matrix of the last N regular-season games for `team`."""
    tmp = df[(df["team"] == team) & (df["season"] <= end_season)].sort_values(["game_date","week"])
    if tmp.empty:
        raise ValueError(f"No rows for {team} up to {end_season}")
    rows = tmp.tail(n).copy()

    # keep only training features, coerce numeric
    X = rows.reindex(columns=feature_names, fill_value=0.0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # If you know the opener venue, you can force is_home here:
    if override_is_home is not None and "is_home" in X.columns:
        X["is_home"] = float(1 if override_is_home else 0)

    return X

def pct(x): return round(float(x)*100, 1)

def main():
    df, model, feats = load_artifacts()

    # Build per-game matrices (not averaged features)
    X_dal = last_n_rows_feature_matrix(df, "DAL", feats, n=12, end_season=2024)
    X_phi = last_n_rows_feature_matrix(df, "PHI", feats, n=12, end_season=2024)

    # Predict on each game, then average the probabilities
    p_dal = float(model.predict_proba(X_dal)[:,1].mean())
    p_phi = float(model.predict_proba(X_phi)[:,1].mean())

    # Normalize to head-to-head split
    total = p_dal + p_phi
    dal_win = (p_dal/total) if total else 0.5
    phi_win = (p_phi/total) if total else 0.5

    result = {
        "Cowboys_Win_Prob": pct(dal_win),
        "Eagles_Win_Prob":  pct(phi_win),
        "method": "avg_of_last_12_game_probs",
        "snapshot_games": 12,
        "seasons_used_for_snapshot": "2022-2024"
    }

    out = OUTPUT_DIR / "cowboys_eagles_2025.json"
    with open(out, "w") as f: json.dump(result, f, indent=2)

    print("✅ Prediction complete")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
