# scripts/02_train_models.py
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib

DATA_PATH = Path("data/processed/team_games_2022_2024.parquet")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate(model, X, y, label):
    prob = model.predict_proba(X)[:,1]
    pred = (prob >= 0.5).astype(int)
    acc   = accuracy_score(y, pred)
    ll    = log_loss(y, prob)
    brier = brier_score_loss(y, prob)
    auc   = roc_auc_score(y, prob)
    print(f"{label}: Acc={acc:.3f}  LogLoss={ll:.3f}  Brier={brier:.3f}  AUC={auc:.3f}")
    return acc, ll, brier, auc

def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    # target
    y = df["win"].copy()

    # feature selection (exclude identifiers / leakage)
    exclude = ["game_id","season","week","team","opponent","game_date","points_for","points_against","win"]
    features = [c for c in df.columns if c not in exclude]

    X = df[features].copy()
    # Ensure numeric-only training (strings -> NaN -> 0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Split by season (train=2022-2023, test=2024)
    train_idx = df["season"].isin([2022, 2023])
    test_idx  = df["season"].isin([2024])

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test,  y_test  = X.loc[test_idx],  y.loc[test_idx]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # OPTIONAL baseline
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    evaluate(logreg, X_train, y_train, "Train LogReg")
    evaluate(logreg, X_test,  y_test,  "Test  LogReg")

    # LightGBM with calibration (isotonic, 5-fold CV)
    base = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    calibrated = CalibratedClassifierCV(base, cv=5, method="isotonic")
    calibrated.fit(X_train, y_train)

    evaluate(calibrated, X_train, y_train, "Train LGBM-Cal")
    evaluate(calibrated, X_test,  y_test,  "Test  LGBM-Cal")

    # Save model + feature list
    joblib.dump(calibrated, MODEL_DIR / "final_model.pkl")
    with open(MODEL_DIR / "features.json", "w") as f:
        json.dump(list(X_train.columns), f)

    # Write simple report
    with open(REPORT_DIR / "metrics.txt", "w") as f:
        for name, model in [("LogReg", logreg), ("LGBM-Cal", calibrated)]:
            acc,ll,brier,auc = evaluate(model, X_test, y_test, f"Test {name}")
            f.write(f"{name}: Acc={acc:.3f}, LogLoss={ll:.3f}, Brier={brier:.3f}, AUC={auc:.3f}\n")

    print("✅ Saved models/final_model.pkl and models/features.json")

if __name__ == "__main__":
    main()  
    
