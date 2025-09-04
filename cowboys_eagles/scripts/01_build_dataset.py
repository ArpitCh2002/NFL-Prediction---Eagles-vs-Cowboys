# scripts/01_build_dataset.py
from pathlib import Path
import pandas as pd
import numpy as np
import nfl_data_py as nfl

# ----------------------------
# CONFIG
# ----------------------------

YEARS = [2022, 2023, 2024]
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

def select_pbp_columns():
    """
    Keep only columns we actually need to reduce memory.
    EPA & success come from nflfastR-derived data (via nfl_data_py).
    """
    return [
        "game_id","season","week",
        "home_team","away_team",
        "posteam","defteam",
        "play_type","pass","rush",
        "epa","success","yards_gained",
        "touchdown","interception","fumble_lost","sack",
    ]

def load_schedules(years):
    # Schedules typically include vegas lines, teams, scores, date/time.
    sched = nfl.import_schedules(years)
    # Keep regular season only (model scope now)
    sched = sched[sched["game_type"] == "REG"].copy()

    keep_cols = [
        "game_id","season","week","gameday","weekday","gametime",
        "home_team","away_team","home_score","away_score",
        # Vegas odds columns (names can be spread_line/total_line or similar)
        "spread_line","total_line","over_under_line",
        # Environment (if present)
        "stadium","roof","surface"
    ]

    # Only keep columns that exist in the dataframe

    keep_cols = [col for col in keep_cols if col in sched.columns]
    sched = sched[keep_cols].copy()
    return sched

def load_pbp(years):
    cols = select_pbp_columns()
    pbp = nfl.import_pbp_data(years, columns=cols)
    # Keep regular season games by referencing schedules later (safer than guessing in PBP)
    return pbp

def aggregate_offense(pbp):
    # Offensive view: group by game_id & posteam
    off = (pbp
           .groupby(["game_id","posteam"], as_index=False)
           .agg(
               plays=("epa","size"),
               epa_offense=("epa","mean"),
               success_offense=("success","mean"),
               pass_rate=("pass","mean"),
               epa_pass=("epa", lambda s: s[pbp.loc[s.index, "pass"]==1].mean()
                         if (pbp.loc[s.index, "pass"]==1).any() else np.nan),
               epa_rush=("epa", lambda s: s[pbp.loc[s.index, "rush"]==1].mean()
                         if (pbp.loc[s.index, "rush"]==1).any() else np.nan),
               turnovers=("interception","sum"),
           ))
    # add fumbles lost to turnovers
    fumbles = (pbp.groupby(["game_id","posteam"], as_index=False)
               .agg(fumbles_lost=("fumble_lost","sum"),
                    sacks_allowed=("sack","sum"),
                    explosive_plays=("yards_gained", lambda s: (s>=20).sum())))
    off = off.merge(fumbles, on=["game_id","posteam"], how="left")
    off["turnovers"] = off["turnovers"].fillna(0) + off["fumbles_lost"].fillna(0)
    return off

def aggregate_defense(pbp):
    # Defensive view: group by game_id & defteam
    deff = (pbp
            .groupby(["game_id","defteam"], as_index=False)
            .agg(
                plays_def=("epa","size"),
                epa_defense=("epa","mean"),           # lower (more negative) is better
                success_against=("success","mean"),
                sacks_made=("sack","sum"),
                takeaways=("interception","sum"),
            ))
    # add fumbles forced (proxied by opponent fumbles lost)
    forced = (pbp.groupby(["game_id","defteam"], as_index=False)
              .agg(fumbles_forced=("fumble_lost","sum")))
    deff = deff.merge(forced, on=["game_id","defteam"], how="left")
    return deff

def build_team_game(pbp, sched):
    off = aggregate_offense(pbp).rename(columns={"posteam":"team"})
    deff = aggregate_defense(pbp).rename(columns={"defteam":"team"})
    team_game = off.merge(deff, on=["game_id","team"], how="left")

    # attach opponent, home/away, scores, odds
    key = sched[["game_id","season","week","gameday","gametime","weekday",
                 "home_team","away_team","home_score","away_score"]].copy()

    # map is_home, opponent, points_for/against
    team_game = team_game.merge(key, on="game_id", how="left")
    team_game["is_home"] = np.where(team_game["team"]==team_game["home_team"], 1,
                             np.where(team_game["team"]==team_game["away_team"], 0, np.nan))
    # opponent
    team_game["opponent"] = np.where(team_game["team"]==team_game["home_team"],
                                     team_game["away_team"], team_game["home_team"])
    # scores
    team_game["points_for"] = np.where(team_game["is_home"]==1,
                                       team_game["home_score"], team_game["away_score"])
    team_game["points_against"] = np.where(team_game["is_home"]==1,
                                           team_game["away_score"], team_game["home_score"])
    team_game["win"] = (team_game["points_for"] > team_game["points_against"]).astype(int)

    # join odds if present
    odds_cols = [c for c in ["spread_line","total_line","over_under_line","stadium","roof","surface"]
                 if c in sched.columns]
    if odds_cols:
        team_game = team_game.merge(sched[["game_id"]+odds_cols], on="game_id", how="left")

    # convert date/time
    if "gameday" in team_game.columns:
        team_game["game_date"] = pd.to_datetime(team_game["gameday"])
    else:
        # fallback if column name differs
        team_game["game_date"] = pd.NaT

    return team_game

def add_rest_and_rolling(df):
    # Sort by date within team so rolling spans across seasons correctly
    df = df.sort_values(["team","game_date","week"]).copy()

    # rest days
    df["rest_days"] = (df.groupby("team")["game_date"].diff().dt.days)

    # rolling windows we want
    roll_feats = [
        "epa_offense","success_offense","epa_pass","epa_rush",
        "epa_defense","success_against","pass_rate",
        "turnovers","sacks_allowed","sacks_made","takeaways","explosive_plays","plays","plays_def"
    ]
    windows = [3, 8, 12]

    for col in roll_feats:
        g = df.groupby("team")[col]
        for w in windows:
            # shift(1) prevents leakage (use only past games)
            df[f"r{w}_{col}"] = g.apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    return df

def main():
    print(f"Loading schedules for {YEARS} ...")
    sched = load_schedules(YEARS)
    sched.to_parquet(RAW_DIR / "schedules_2022_2024.parquet", index=False)
    print(f"Schedules shape: {sched.shape}")

    print(f"Loading play-by-play for {YEARS} ... (this can take a bit)")
    pbp = load_pbp(YEARS)
    # align to regular season by joining present game_ids in schedules
    pbp = pbp[pbp["game_id"].isin(set(sched["game_id"]))]
    pbp.to_parquet(RAW_DIR / "pbp_2022_2024.parquet", index=False)
    print(f"PBP shape (REG only): {pbp.shape}")

    print("Aggregating to team-game level ...")
    team_game = build_team_game(pbp, sched)

    # sanity checks
    assert team_game["team"].notna().all()
    assert team_game.groupby("game_id")["team"].nunique().between(2, 2).all(), \
        "Every game_id should have exactly 2 team rows"

    print("Engineering rest + rolling 3-game features ...")
    team_game = add_rest_and_rolling(team_game)

    # Minimal essential columns for modeling
    essential_cols = [
        "game_id","season","week","team","opponent","is_home","game_date",
        "points_for","points_against","win",
        "spread_line","total_line","over_under_line"
    ]
    essential_cols = [c for c in essential_cols if c in team_game.columns]

    # Order columns: essentials first, then features
    feature_cols = [c for c in team_game.columns
                    if c not in essential_cols + [
                        "home_team","away_team","home_score","away_score","gameday","gametime","weekday"
                    ]]
    ordered = essential_cols + feature_cols
    team_game = team_game[ordered].copy()

    out_path = PROC_DIR / "team_games_2022_2024.parquet"
    team_game.to_parquet(out_path, index=False)

    print(f"Saved Processed Dataset: {out_path}")
    print(team_game.sample(5, random_state=42))


if __name__ == "__main__":
    main()
