# Architecture

This document explains how data moves from the source to the screen, what every folder is for,
and why the main decisions were made. Read it top to bottom once; later use it as a map.

## 1. The one-paragraph version

Open-Meteo serves hourly air-quality and weather data for points in Delhi. `pipeline/ingest.py`
downloads it, validates it and stores it idempotently in a database. `pipeline/aqi.py` turns
concentrations into the Indian National AQI. `pipeline/features.py` builds leakage-safe model
inputs. `ml/` trains a LightGBM quantile forecaster, scores it by walk-forward validation against
two baselines, and writes a 72-hour forecast with an uncertainty band into the database. A FastAPI
backend (`backend/`) serves everything as JSON, and also serves the built Next.js website
(`frontend/`) so the whole product is one process.

## 2. Data flow

```
 Open-Meteo Air Quality (CAMS)  ─┐
 Open-Meteo Weather (ERA5)      ─┼─►  pipeline/ingest.py ──validate, dedupe, upsert──►  DATABASE
 OpenAQ v3 stations (optional)  ─┘      (hourly job)                                    │
                                                                                        │ observations
          pipeline/aqi.py  (NAQI) ◄─────────────────────────────────────────────────────┤
          pipeline/features.py  (features at t use data ≤ t only) ◄─────────────────────┤
                                    │
                  ml/train.py ── ml/evaluate.py (walk-forward) ──► ml/registry/metrics.json (+DB)
                                    │
                  ml/forecast.py ──► forecasts table (issued_at, q10/q50/q90 for PM2.5, PM10, AQI)
                                                                                        │
                      backend/app/service.py  (all business logic)  ◄───────────────────┘
                                    │  JSON /api/*
                      frontend/ (Next.js static export, no logic)  ──►  browser
```

The hourly job (`python -m pipeline.jobs refresh`) is the only thing that writes. The API only
reads, so a web request never trains a model or calls an upstream service (except the optional,
cached OpenAQ lookup).

## 3. Source of truth

| Question | Answer |
|---|---|
| What series does the forecaster learn and predict? | Open-Meteo Air Quality (the CAMS global model), hourly, from 2022-08-04. |
| Why not ground sensors? | They have gaps and faults, need an API key, and have short or patchy history. They are an optional overlay. |
| What is the catch? | It is a model on a ~0.4° grid, not a measurement. Inside Delhi it resolves only two independent cells. The README's limitations section says so, and so does the website. |
| Where do weather inputs come from? | ERA5 archive (complete until ~6 days ago) plus the forecast endpoint for the most recent days. Only past weather is used as a feature. |
| Is anything revised later? | Yes. The newest hours are CAMS forecast runs, replaced when analysis arrives. `update` re-fetches the last 72 hours so revisions land. |

## 4. Folder map

```
delhi-aqi/
  run.py                  one-command launcher (venv + install + bootstrap + server + browser)
  README.md  LICENSE  .env.example  Makefile  pyproject.toml
  requirements.txt  requirements-dev.txt
  Dockerfile  docker-compose.yml  .dockerignore  render.yaml
  .github/workflows/      ci.yml (lint, tests, frontend build, docker) · refresh.yml (hourly cron job)

  pipeline/               "get data and make it usable"
    config.py               every setting, one place (env vars / .env)
    locations.py            the 2 model cells, the city average, 21 neighbourhoods mapped to cells
    db.py                   schema (SQLite or PostgreSQL), upserts, read helpers
    ingest.py               fetch → validate → store; backfill / update / status CLI
    aqi.py                  CPCB NAQI maths, categories, averaging windows
    features.py             leakage-safe features, calendar/Diwali flags, (origin, horizon) rows
    openaq.py               optional real stations (needs OPENAQ_API_KEY)
    jobs.py                 bootstrap / refresh / train / forecast / worker commands

  ml/                     "learn and predict"
    baselines.py            persistence and seasonal naive (every model must beat these)
    train.py                LightGBM quantile forecaster, conformal calibration, SARIMA
    evaluate.py             WALK-FORWARD evaluation only; MAE/RMSE per horizon, intervals
    forecast.py             train the production model, produce the live 72 h forecast
    registry.py             where models live (DB table); registry/metrics.json is the shipped evaluation

  backend/app/            "serve it"
    main.py                 app factory, CORS, error handlers, serves the built site
    service.py              business logic behind every endpoint
    routes/                 thin routers: /health /current /history /seasonal /forecast /stations /model-metrics
    scheduler.py            optional in-process hourly refresh (single-container runs)
    static/                 the built website (committed so end users need no Node)

  frontend/               "show it" (Next.js + TypeScript, static export)
    src/app/                page.tsx (dashboard), model/page.tsx (model performance)
    src/components/         AqiCard, ForecastChart, HistoryChart, SeasonalHeatmap, StationMap, Header
    src/lib/api.ts          the only code that talks to the backend; typed responses

  data/                   local SQLite file and downloads (git-ignored; fetched, never committed)
  notebooks/01_eda.ipynb  executed exploratory analysis with real findings
  scripts/                check_grid_cells.py · build_eda_notebook.py · build_frontend.py
  tests/                  pytest suite: offline, synthetic data, runs in seconds
  docs/                   this file
```

## 5. The database

SQLite by default (a file in `data/`), PostgreSQL when `DATABASE_URL` says so. The same SQLAlchemy
schema serves both, and `upsert()` uses each dialect's `ON CONFLICT DO UPDATE`. All timestamps are
naive UTC.

| Table | Key | Purpose |
|---|---|---|
| `observations` | `(location_id, ts)` | hourly pollutants and weather per location |
| `forecasts` | `(location_id, issued_at, target_ts)` | q10/q50/q90 of PM2.5, PM10 and AQI for horizons 1..72 |
| `locations` | `id` | the tracked places |
| `ingest_log` | `id` | every ingestion run: status, rows, data-through time (this feeds "data freshness") |
| `model_artifacts` | `name` | the trained model (gzip JSON of LightGBM text models) |
| `eval_results` | `id` | the latest walk-forward evaluation |
| `meta` | `key` | small key/value facts (last forecast time, ...) |

## 6. The three ideas that make the forecasting honest

**No leakage.** A forecast *origin* `t` is the last observed hour. Every input is computed from data
at or before `t` (`shift(k ≥ 0)` and trailing windows only). Calendar features describe the target
hour and carry no observations. `tests/test_features.py` destroys everything after `t` and asserts
that no feature at `t` changes, and also proves that the test would catch a leaky feature.

**Walk-forward only.** `ml/evaluate.py` picks cut-offs across the history. For each, it trains on
rows whose *target time* is before the cut-off, so no training example overlaps the test period,
and tests on the following 30 days. There is no random split anywhere in the repository.

**Beat the baselines, show the uncertainty.** Persistence ("nothing changes") and seasonal naive
("same hour yesterday") are scored on exactly the same hours as the model, and every table on the
model page shows all of them. The forecast is a 10th/50th/90th percentile band, calibrated on
held-out data (split-conformal), and the evaluation reports how often the truth really fell inside
the band.

## 7. The model

One *global* LightGBM model per (target, quantile) serves every location and every horizon:
`horizon` and `loc_code` are inputs. That is 2 targets × 3 quantiles = 6 boosters. Each predicts
`log1p(y[t+h]) − log1p(y[t])`, the change from now in log space, so a forecast starts at
persistence and the trees learn the correction (quantiles survive that monotone transform).
Conformal calibration widens or narrows the 10–90 band per horizon bucket using the most recent 120
days of held-out targets, then the models are refitted on all data.

Why not an LSTM or Prophet? On roughly 36 000 hourly rows the gradient-boosted model is competitive,
trains in seconds, gives quantiles natively and is easy to explain. An LSTM is a reasonable
stretch goal; it would need its own walk-forward run and a heavier image.

## 8. AQI basis

`AQI_BASIS=pm2_5` (default), `pm` or `all`. The official NAQI takes the maximum sub-index over six
pollutants; `all` implements that exactly. On this model data, ozone becomes the "dominant" pollutant
in 53 % of hours and PM10 shows a level change from March 2025 (see the EDA notebook), so the
default headline uses PM2.5. Every pollutant's sub-index is still reported on the dashboard.

## 9. Running it three ways

| Way | Command | Database | Scheduler |
|---|---|---|---|
| Plain Python | `python run.py` | SQLite file | in-process, hourly |
| Docker Compose | `docker compose up --build` | PostgreSQL container | separate `worker` service |
| Cloud (free tier) | see README → Deploy | external PostgreSQL | in-process + GitHub Actions cron |

## 10. Security and configuration

* Secrets are environment variables only; `.env` is git-ignored, `.env.example` is committed.
* The API is read-only (GET) and has no user input that reaches SQL except bounded, validated query parameters.
* The Docker image runs as a non-root user.

## 11. Testing

`pytest` runs offline in under a minute. Highlights: golden values for the NAQI, the leakage tests,
the walk-forward "no training target after the cut-off" test, an end-to-end train → forecast → API
test on synthetic data, mocked-HTTP ingestion tests (retry, idempotency, never storing future
hours) and a test for each API route. CI runs lint, tests, the frontend type-check and build, and
validates the Compose file.
