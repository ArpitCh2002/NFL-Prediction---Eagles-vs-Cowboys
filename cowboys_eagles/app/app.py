# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, base64
from pathlib import Path
import streamlit.components.v1 as components

# -----------------------
# Paths
# -----------------------
DATA_PATH = Path("data/processed/team_games_2022_2024.parquet")
MODEL_PATH = Path("models/final_model.pkl")
FEATURES_PATH = Path("models/features.json")
LOGO_DIR = Path("app/static/logos")  # put DAL.png, PHI.png, etc. here

# -----------------------
# Caching
# -----------------------
@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)

@st.cache_resource
def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    feats = json.load(open(FEATURES_PATH))
    return model, feats

# -----------------------
# Helpers
# -----------------------
def last_n_rows_feature_matrix(
    df: pd.DataFrame,
    team: str,
    feature_names,
    n: int = 12,
    end_season: int = 2024,
    override_is_home: bool | None = None
) -> pd.DataFrame:
    """
    Return an aligned numeric matrix of the last N regular-season games for `team`,
    up to and including `end_season`. Optionally override is_home for all rows.
    """
    tmp = df[(df["team"] == team) & (df["season"] <= end_season)].sort_values(["game_date","week"])
    if tmp.empty:
        raise ValueError(f"No rows for {team} up to season {end_season}")
    rows = tmp.tail(n).copy()

    X = rows.reindex(columns=feature_names, fill_value=0.0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    if override_is_home is not None and "is_home" in X.columns:
        X["is_home"] = float(1 if override_is_home else 0)

    return X

def to_percent(p: float, decimals: int = 1) -> float:
    return round(float(p) * 100.0, decimals)

def clip_width(pct: float) -> float:
    """Avoid visual clipping at 0/100 in the bar; keep between 2 and 98."""
    return float(max(2.0, min(98.0, pct)))

def read_logo_b64(team_code: str) -> str | None:
    """
    Loads team logo as base64 data URL if found. Looks for PNG then JPG/JPEG.
    Returns a 'data:image/...;base64,...' string or None if not found.
    """
    candidates = [
        LOGO_DIR / f"{team_code}.png",
        LOGO_DIR / f"{team_code}.jpg",
        LOGO_DIR / f"{team_code}.jpeg",
    ]
    for p in candidates:
        if p.exists():
            data = p.read_bytes()
            mime = "image/png" if p.suffix.lower()==".png" else "image/jpeg"
            return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"
    return None

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="NFL Prediction Card", page_icon="🏈", layout="centered")
st.title("🏈 NFL Prediction Card")
st.caption("Predicts using last-N games per team across 2022–2024, with a calibrated model. Download the exact card as PNG.")

# Load data + model
df = load_data()
model, feature_names = load_model_and_features()

# Team selectors
teams = sorted(df["team"].unique())
c1, c2 = st.columns(2)
team1 = c1.selectbox("Team 1", teams, index=teams.index("DAL") if "DAL" in teams else 0)
team2 = c2.selectbox("Team 2", teams, index=teams.index("PHI") if "PHI" in teams else 1)

# NEW: display names for the card
TEAM_LABELS = {"DAL": "Cowboys", "PHI": "Eagles"}
label1 = TEAM_LABELS.get(team1, team1)
label2 = TEAM_LABELS.get(team2, team2)

# Controls row
c3, c4, c5 = st.columns([1, 1, 2])
last_n = c3.slider("Last N games", min_value=6, max_value=16, value=12, step=1)
end_season = c4.selectbox("Use games up to season", options=[2024, 2023, 2022], index=0)
kickoff_text = c5.text_input("Kickoff text on the card", "Mon • Sep 8 • 8:20 PM ET")

# Venue override (optional)
venue_opt = st.selectbox(
    "Venue override (optional)",
    options=["Auto (use past rows)", "Team 1 is HOME", "Team 2 is HOME"],
    index=0
)
if venue_opt == "Team 1 is HOME":
    override_home_1, override_home_2 = True, False
elif venue_opt == "Team 2 is HOME":
    override_home_1, override_home_2 = False, True
else:
    override_home_1 = override_home_2 = None

# Manual override option
manual = st.checkbox("Enter probabilities manually (skip model)")

# Guard: distinct teams
if team1 == team2:
    st.warning("Please select two different teams.")
    st.stop()

# Compute probabilities
if manual:
    p1_pct = float(st.slider(f"{team1} win probability (%)", 0, 100, 55))
    p2_pct = round(100 - p1_pct, 1)
else:
    try:
        X1 = last_n_rows_feature_matrix(df, team1, feature_names, n=last_n, end_season=end_season, override_is_home=override_home_1)
        X2 = last_n_rows_feature_matrix(df, team2, feature_names, n=last_n, end_season=end_season, override_is_home=override_home_2)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Predict on each of the last-N games and average probabilities
    prob1 = float(model.predict_proba(X1)[:, 1].mean())
    prob2 = float(model.predict_proba(X2)[:, 1].mean())

    # Normalize to head-to-head
    total = prob1 + prob2
    p1 = (prob1 / total) if total else 0.5
    p2 = (prob2 / total) if total else 0.5

    p1_pct = to_percent(p1, 1)
    p2_pct = to_percent(p2, 1)

# Determine colors: higher % = green, lower % = red
GREEN = "#22c55e"
RED   = "#ef4444"
if p1_pct >= p2_pct:
    bar1_color, bar2_color = GREEN, RED
else:
    bar1_color, bar2_color = RED, GREEN

# Bar widths (visual only)
bar1 = clip_width(p1_pct)
bar2 = clip_width(p2_pct)

# Logos (base64 or fallback dot)
team1_logo = read_logo_b64(team1)
team2_logo = read_logo_b64(team2)
if team1_logo:
    team1_logo_html = f'<img class="logo-img" src="{team1_logo}" alt="{team1} logo" />'
else:
    team1_logo_html = '<div class="logo"></div>'
if team2_logo:
    team2_logo_html = f'<img class="logo-img" src="{team2_logo}" alt="{team2} logo" />'
else:
    team2_logo_html = '<div class="logo" style="background: radial-gradient(circle at 30% 30%, #f97316 30%, #c2410c 65%, #7c2d12 100%);"></div>'

# -----------------------
# Card HTML/CSS/JS
# -----------------------
card_html = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700&display=swap" rel="stylesheet">

<style>
  .card-wrap {{
    display:flex; justify-content:center; margin: 8px 0 24px 0;
  }}
  .card {{
    width: 460px;
    background: linear-gradient(180deg, #6b7cff 0%, #6e79f6 100%);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.18);
    font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji';
    color: #0f172a;
    position: relative;
  }}
  .card-inner {{
    background: #ffffff;
    border-radius: 14px;
    padding: 18px 18px 14px 18px;
    position: relative;
  }}
  /* Moved the time pill DOWN ~4-5px inside the card */
  .time-pill {{
    position: absolute;
    top: 2px;              /* was -12px */
    right: 12px;
    background: #7b8cff;
    color: #fff;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(0,0,0,0.15);
  }}
  /* Align the green dot with the new pill position */
  .status-dot {{
    position: absolute;
    top: 10px;             /* was -6px */
    right: 10px;
    width: 10px; height: 10px;
    background: #22c55e;
    border-radius: 50%;
    border: 2px solid #fff;
  }}
  .row {{
    display:flex; align-items:center; gap:12px; margin: 10px 0 14px 0;
  }}
  /* Fallback dot */
  .logo {{
    width: 28px; height: 28px; border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #ea3a3a 30%, #b00000 65%, #8a0000 100%);
    box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
    flex: 0 0 28px;
  }}
  /* Real logo image */
  .logo-img {{
    width: 28px; height: 28px; border-radius: 50%;
    object-fit: cover;
    border: 1px solid rgba(0,0,0,0.06);
    flex: 0 0 28px;
  }}
  .right {{
    flex: 1; display:flex; align-items:center; gap:10px;
  }}
  .team-name {{
    font-weight: 700; color: #111827; font-size: 14px; min-width: 140px;
  }}
  .bar-wrap {{
    flex: 1;
    background: #eef2ff;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    width: 50%;
    transition: width 300ms ease;
  }}
  .pct {{
    width: 52px; text-align:right; font-weight: 700; color: #111827; font-size: 13px;
  }}
  .vs-chip {{
    display:inline-block;
    margin: 0 0 4px 40px;
    padding: 4px 10px;
    background: #eef2ff;
    color: #4f46e5;
    border-radius: 999px;
    font-weight: 700; font-size: 12px;
  }}
</style>

<div class="card-wrap">
  <div class="card">
    <div class="card-inner" id="predict-card">
      <div class="time-pill">{kickoff_text}</div>
      <div class="status-dot"></div>

      <div class="row">
        {team1_logo_html}
        <div class="right">
          <div class="team-name">{label1}</div>
          <div class="bar-wrap">
            <div class="bar-fill" style="width:{bar1}%; background:{bar1_color};"></div>
          </div>
          <div class="pct">{p1_pct:.1f}%</div>
        </div>
      </div>

      <div class="row">
        {team2_logo_html}
        <div class="right">
          <div class="team-name">{label2}</div>
          <div class="bar-wrap">
            <div class="bar-fill" style="width:{bar2}%; background:{bar2_color};"></div>
          </div>
          <div class="pct">{p2_pct:.1f}%</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Download button (client-side) using html2canvas -->
<div style="display:flex; justify-content:center;">
  <button id="dl-btn" style="
    background:#111827; color:#fff; padding:10px 14px; border:none; border-radius:10px;
    font-weight:700; cursor:pointer; box-shadow:0 6px 14px rgba(0,0,0,0.15);">
    Download Card as PNG
  </button>
</div>

<script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
<script>
  const btn = document.getElementById('dl-btn');
  btn.addEventListener('click', async () => {{
    const node = document.getElementById('predict-card');
    const canvas = await html2canvas(node, {{backgroundColor: null, scale: 2}});
    const link = document.createElement('a');
    link.download = '{team1}_vs_{team2}_card.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
  }});
</script>
"""

components.html(card_html, height=340, scrolling=False)
st.caption("Logos: add PNGs in app/static/logos/ named by team code (e.g., DAL.png, PHI.png). Kickoff pill moved inside the card; bars: green = higher %, red = lower %.")
