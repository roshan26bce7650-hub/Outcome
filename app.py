from pathlib import Path
import io
import json
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = DATA_DIR / "trained_model.joblib"
META_PATH = DATA_DIR / "model_meta.json"
DATASET_PATH = DATA_DIR / "games.csv"
DATA_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# Supported starter catalogue. Add games in this list as the product expands.
GAME_CATALOG = [
    {"name":"Cricket","category":"Sports","icon":"🏏","description":"International, league and domestic matches","examples":"India, Australia, IPL teams"},
    {"name":"Football","category":"Sports","icon":"⚽","description":"Association football and club competitions","examples":"Real Madrid, Barcelona, Brazil"},
    {"name":"Basketball","category":"Sports","icon":"🏀","description":"Professional and international basketball","examples":"Lakers, Celtics, Warriors"},
    {"name":"Tennis","category":"Sports","icon":"🎾","description":"Singles and doubles matchups","examples":"Player A, Player B"},
    {"name":"American Football","category":"Sports","icon":"🏈","description":"League and college matchups","examples":"Chiefs, Eagles, 49ers"},
    {"name":"Baseball","category":"Sports","icon":"⚾","description":"League and international baseball","examples":"Yankees, Dodgers"},
    {"name":"Ice Hockey","category":"Sports","icon":"🏒","description":"League and international hockey","examples":"Oilers, Rangers"},
    {"name":"Rugby","category":"Sports","icon":"🏉","description":"Rugby union and league","examples":"New Zealand, South Africa"},
    {"name":"Badminton","category":"Sports","icon":"🏸","description":"Singles and doubles matchups","examples":"Player A, Player B"},
    {"name":"Volleyball","category":"Sports","icon":"🏐","description":"Club and international volleyball","examples":"Team A, Team B"},
    {"name":"League of Legends","category":"Esports","icon":"🧙","description":"MOBA competitive matches","examples":"T1, Gen.G, G2"},
    {"name":"Counter-Strike 2","category":"Esports","icon":"🎯","description":"Tactical FPS competitive matches","examples":"Team Vitality, FaZe"},
    {"name":"Dota 2","category":"Esports","icon":"🛡️","description":"MOBA competitive matches","examples":"Team Spirit, OG"},
    {"name":"Valorant","category":"Esports","icon":"🔻","description":"Tactical FPS competitive matches","examples":"Fnatic, Sentinels"},
    {"name":"PUBG / BGMI","category":"Esports","icon":"🪖","description":"Battle royale esports","examples":"Squad A, Squad B"},
    {"name":"Apex Legends","category":"Esports","icon":"🚀","description":"Battle royale esports","examples":"Squad A, Squad B"},
    {"name":"Rainbow Six Siege","category":"Esports","icon":"🕹️","description":"Tactical team shooter","examples":"Team A, Team B"},
    {"name":"Overwatch 2","category":"Esports","icon":"🦸","description":"Team-based competitive shooter","examples":"Team A, Team B"},
    {"name":"Rocket League","category":"Esports","icon":"🚗","description":"Competitive car football","examples":"Team BDS, Vitality"},
    {"name":"Fighting Games","category":"Esports","icon":"🥊","description":"Street Fighter, Tekken and more","examples":"Player A, Player B"},
    {"name":"Other / Custom","category":"Other","icon":"✳️","description":"Use the same feature schema for another game","examples":"Any two competitors"},
]

# Standardized CSV schema used by the trainer. Each row is a historical matchup.
FEATURE_COLUMNS = [
    "game", "participant_a", "participant_b",
    "a_win_rate", "b_win_rate",
    "a_recent_form", "b_recent_form",
    "a_rank", "b_rank",
    "a_rest_days", "b_rest_days",
    "a_home_advantage"
]
TARGET_COLUMN = "outcome"
NUMERIC_FEATURES = [
    "a_win_rate", "b_win_rate", "a_recent_form", "b_recent_form",
    "a_rank", "b_rank", "a_rest_days", "b_rest_days", "a_home_advantage"
]
CATEGORICAL_FEATURES = ["game"]
VALID_OUTCOMES = {"A", "B", "DRAW"}

def load_meta():
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"trained": False, "message": "No real dataset has been trained yet."}

def build_pipeline():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ))
    ])

def clean_dataset(df):
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    df = df[required].copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["game", "participant_a", "participant_b", TARGET_COLUMN]:
        df[col] = df[col].astype(str).str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.upper()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["game", TARGET_COLUMN])
    df = df[df[TARGET_COLUMN].isin(VALID_OUTCOMES)]
    df = df.dropna(subset=["participant_a", "participant_b"])
    df = df[(df["participant_a"] != "") & (df["participant_b"] != "")]
    if len(df) < 20:
        raise ValueError("Please provide at least 20 valid historical rows. More data is strongly recommended.")
    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("The dataset must contain at least two different outcome classes (A, B, DRAW).")
    for col in ["a_win_rate", "b_win_rate", "a_recent_form", "b_recent_form"]:
        df[col] = df[col].clip(0, 1)
    for col in ["a_rank", "b_rank", "a_rest_days", "b_rest_days"]:
        df[col] = df[col].clip(lower=0)
    df["a_home_advantage"] = df["a_home_advantage"].clip(0, 1)
    return df

def train_from_dataframe(df):
    df = clean_dataset(df)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    joblib.dump(model, MODEL_PATH)
    DATASET_PATH.write_text(df.to_csv(index=False), encoding="utf-8")
    meta = {
        "trained": True,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "test_rows": int(len(y_test)),
        "accuracy": round(accuracy, 4),
        "classes": list(model.named_steps["classifier"].classes_),
        "report": report,
        "note": "Accuracy is measured on a random held-out split. Real-world performance may differ; use time-based validation for time-ordered data."
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

@app.get("/")
def index():
    return render_template("index.html", games=GAME_CATALOG)

@app.get("/api/status")
def api_status():
    meta = load_meta()
    return jsonify({
        "trained": bool(meta.get("trained")),
        "meta": meta,
        "games": GAME_CATALOG,
        "features": FEATURE_COLUMNS,
        "outcomes": sorted(VALID_OUTCOMES)
    })

@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    game = str(payload.get("game", "Other / Custom")).strip()
    a = str(payload.get("participant_a", "Competitor A")).strip()[:80]
    b = str(payload.get("participant_b", "Competitor B")).strip()[:80]
    if not a or not b or a.lower() == b.lower():
        return jsonify({"error": "Enter two different competitor or team names."}), 400
    try:
        row = {
            "game": game,
            "participant_a": a,
            "participant_b": b,
            "a_win_rate": float(payload.get("a_win_rate", 0.5)),
            "b_win_rate": float(payload.get("b_win_rate", 0.5)),
            "a_recent_form": float(payload.get("a_recent_form", 0.5)),
            "b_recent_form": float(payload.get("b_recent_form", 0.5)),
            "a_rank": float(payload.get("a_rank", 50)),
            "b_rank": float(payload.get("b_rank", 50)),
            "a_rest_days": float(payload.get("a_rest_days", 3)),
            "b_rest_days": float(payload.get("b_rest_days", 3)),
            "a_home_advantage": float(payload.get("a_home_advantage", 0))
        }
    except (TypeError, ValueError):
        return jsonify({"error": "All statistics must be valid numbers."}), 400
    for key in ["a_win_rate", "b_win_rate", "a_recent_form", "b_recent_form", "a_home_advantage"]:
        if not 0 <= row[key] <= 1:
            return jsonify({"error": f"{key.replace('_', ' ')} must be between 0 and 1 (0–100%)."}), 400
    for key in ["a_rank", "b_rank", "a_rest_days", "b_rest_days"]:
        if row[key] < 0:
            return jsonify({"error": f"{key.replace('_', ' ')} cannot be negative."}), 400

    if not MODEL_PATH.exists() or not load_meta().get("trained"):
        return jsonify({
            "error": "A real prediction model is not trained yet. Upload historical results in the Model Lab first.",
            "needs_training": True
        }), 409

    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([{k: row[k] for k in FEATURE_COLUMNS}])
    model_start = time.perf_counter()
    probabilities = model.predict_proba(X)[0]
    classes = list(model.named_steps["classifier"].classes_)
    elapsed_ms = round((time.perf_counter() - model_start) * 1000, 2)
    output = {c: float(p) for c, p in zip(classes, probabilities)}
    # Missing classes are shown as 0 for a consistent UI.
    for c in VALID_OUTCOMES:
        output.setdefault(c, 0.0)
    winner = max(output, key=output.get)
    display_names = {"A": a, "B": b, "DRAW": "Draw"}
    return jsonify({
        "game": game,
        "participant_a": a,
        "participant_b": b,
        "probabilities": [
            {"outcome": "A", "label": a, "probability": round(output["A"] * 100, 1)},
            {"outcome": "B", "label": b, "probability": round(output["B"] * 100, 1)},
            {"outcome": "DRAW", "label": "Draw", "probability": round(output["DRAW"] * 100, 1)}
        ],
        "top_outcome": display_names[winner],
        "top_probability": round(output[winner] * 100, 1),
        "latency_ms": elapsed_ms,
        "model_accuracy": load_meta().get("accuracy"),
        "is_real_model": True,
        "disclaimer": "Probabilities are model estimates, not guarantees. Prediction quality depends on dataset quality and relevance."
    })

@app.post("/api/train")
def api_train():
    # No login by requirement. This is a demo-friendly admin route; protect it with auth/API key before public production.
    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename.lower().endswith(".csv"):
        return jsonify({"error": "Choose a CSV file."}), 400
    try:
        df = pd.read_csv(io.BytesIO(uploaded.read()))
        meta = train_from_dataframe(df)
        return jsonify({"ok": True, "meta": meta})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.get("/api/template.csv")
def api_template():
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    return app.response_class(
        ",".join(columns) + "\n",
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=game_outcomes_template.csv"}
    )

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
