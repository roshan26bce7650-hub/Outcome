# Outcome — Game Outcome Prediction Lab

A responsive Flask web app with a searchable sports/esports directory, no-login matchup builder, prediction dashboard, CSV model-training panel, and a scikit-learn Random Forest pipeline.

## Important model/data note

The directory includes popular sports and esports titles, but the app does **not** invent predictions before a model is trained. Upload a historical dataset in the Model Lab. The model then returns probability estimates based on the supplied features. A game appearing in the directory does not mean the model has reliable data for it.

The held-out score is a basic random split. For a serious predictive product, validate chronologically, benchmark against a baseline, and report metrics separately for each game/competition. Never represent model probabilities as guarantees.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Train the model

1. Open **Model lab** in the site.
2. Download the CSV template.
3. Add at least 20 valid historical match rows (more is strongly recommended).
4. Upload the CSV and select **Train model**.
5. Once training succeeds, return to the Predictor and enter the same feature definitions.

### CSV schema

Required columns:

`game,participant_a,participant_b,a_win_rate,b_win_rate,a_recent_form,b_recent_form,a_rank,b_rank,a_rest_days,b_rest_days,a_home_advantage,outcome`

- `game`: game/sport name or competition category.
- `participant_a`, `participant_b`: names for the historical matchup.
- `a_win_rate`, `b_win_rate`: pre-match historical win rate from `0` to `1`.
- `a_recent_form`, `b_recent_form`: recent form rate from `0` to `1`.
- `a_rank`, `b_rank`: numeric ratings/ranks on a consistent scale. If using rank numbers, lower may mean better; be consistent.
- `a_rest_days`, `b_rest_days`: rest days before the match.
- `a_home_advantage`: `1` if A has home advantage, otherwise `0`.
- `outcome`: `A`, `B`, or `DRAW`.

The same definitions and time window must be used for training and live predictions. Use only information that would be known before the match to avoid data leakage.

## Deploy to Render

This repository includes `render.yaml` and `Procfile`.

1. Push the project to a GitHub repository.
2. In Render, create a new Blueprint or Web Service from the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Deploy and open the generated Render URL.

### Persistence warning

Render's local filesystem may be ephemeral. For a production deployment, use a persistent disk or external object/database storage for uploaded datasets and the trained model. Otherwise, retraining or deploys may lose the model.

## Security warning before public release

The user requested no login, so prediction access is public. **The CSV model-training endpoint is also public in this demo.** Do not expose it to the public internet as-is: anyone could retrain or replace the model. Add admin authentication/API-key protection, request limits, audit logs, file validation, and persistent storage before production. No-login for users can coexist with a separately protected admin route.

## Project structure

```text
game_outcome_predictor/
├── app.py
├── requirements.txt
├── Procfile
├── render.yaml
├── README.md
├── data/
├── static/
│   ├── app.js
│   └── style.css
└── templates/
    └── index.html
```

## API endpoints

- `GET /api/status` — model status and game catalogue.
- `POST /api/predict` — validate inputs and return probabilities if a trained model exists.
- `POST /api/train` — train/retrain from a CSV upload (protect before public deployment).
- `GET /api/template.csv` — download the CSV header template.
- `GET /health` — deployment health check.
