# Delhi AQI

Live-style air-quality dashboard for Delhi: **current AQI** on the Indian National AQI scale, a **72-hour
probabilistic forecast** with an uncertainty band, **historical trends**, a **seasonal heatmap**, a **map**, and a
**model-performance page** that scores every forecast honestly against simple baselines.

![Dashboard](docs/img/dashboard.png)

> **Live demo:** not deployed from this repository yet. Deploying needs your Render/Neon accounts, so see
> [Deploy](#deploy) for the steps. Everything else runs locally with one command.

---

## Quick start (one command)

You need **Python 3.11 or newer** and an **internet connection**. Nothing else: no Node, no Docker, no database.

```bash
python run.py          # on Windows you can also use:  py run.py
```

That single command creates a virtual environment, installs the dependencies, downloads about four years of hourly
data, trains the models, makes the first forecast, starts the server and opens <http://127.0.0.1:8000>.

* **First run: roughly 5 to 10 minutes** (download + install + training). The page shows a "setup is running" message
  if you open it earlier. Every later run starts in seconds.
* While it runs it refreshes the data and forecast **every hour** by itself.
* Stop with `Ctrl+C`. Your data stays in `data/delhi_aqi.db`.

Useful options: `python run.py --no-browser`, `--port 8080`, `--reinstall`.

**Nothing else to configure.** No API keys are needed. Everything optional is listed under [Configuration](#configuration).

---

## What the numbers say (walk-forward evaluation on real data)

12 walk-forward folds, 30-day test windows, forecast origins every 6 hours, Delhi city-average PM2.5, data from
August 2022 to October 2026. Every model is scored on **exactly the same hours**. MAE in µg/m³, lower is better.

| Hours ahead | Persistence ("nothing changes") | Seasonal naive ("same hour yesterday") | **LightGBM (this project)** | Improvement over persistence |
|---:|---:|---:|---:|---:|
| 1 | 7.2 | 25.0 | **7.1** | tie |
| 3 | 17.6 | 24.6 | **12.4** | 30 % |
| 6 | 25.2 | 25.3 | **18.3** | 28 % |
| 12 | 32.6 | 25.3 | **21.3** | 35 % |
| 24 | 25.2 | 25.2 | **22.6** | 10 % |
| 48 | 30.7 | 30.7 | **26.5** | 13 % |
| 72 | 32.2 | 32.2 | **27.2** | 15 % |

* The model beats persistence at **every** horizon and in **12 of 12** folds, but the edge at 1 hour is nil and shrinks
  to 10 to 15 % beyond a day. Beyond 24 hours the forecast is a trend with a wide band, not a prediction.
* The **80 % prediction interval really covers 79 to 84 %** of outcomes (raw quantile regression covered only 68 to 77 %
  before conformal calibration).
* PM10 is harder: the model is *worse* than persistence at 1 hour and better from 3 hours onward.
* A classical **SARIMA** is competitive: it beats the LightGBM model at 72 hours on the subset where it was run
  (28.1 vs 30.5 µg/m³). The model page shows this table unedited.
* Winter (Nov to Feb) errors are larger in absolute terms, and the band stays slightly conservative there.

The same tables, charts and the full method are on the website's **Model performance** page, and in
`ml/registry/metrics.json`. Re-run them yourself with `make evaluate` (about 10 minutes).

![Model page](docs/img/model.png)

---

## Data sources and what is the source of truth

| Source | Used for | Key? |
|---|---|---|
| **Open-Meteo Air Quality API** (CAMS model) | **The series the models learn and forecast** (PM2.5, PM10, NO2, SO2, CO, O3, hourly, from 2022-08-04) | none |
| Open-Meteo Weather (ERA5 archive + forecast endpoint) | Weather inputs (boundary-layer height, wind, humidity, ...) | none |
| OpenAQ v3 | *Optional* real ground sensors on the map | free key |
| CPCB / Kaggle datasets | Not used by the app (useful for offline validation against sensors) | token / manual |

Open-Meteo is free for non-commercial use under **CC BY 4.0** (credit is shown on the site). Check their terms before any
commercial use.

### Things I found in the data that you should know (see `notebooks/01_eda.ipynb`)

1. **Delhi is only two independent model cells.** The CAMS grid is so coarse that 21 neighbourhoods return exactly
   identical series inside a cell (`python scripts/check_grid_cells.py` reproduces this). So the app forecasts the
   *north/central cell, the south cell and their average*, and the map colours neighbourhoods by their cell. **Per-station
   forecasts from this source would be fiction.** Real per-sensor data needs OpenAQ.
2. **Ozone and dust are unreliable in this model.** Counting all six pollutants makes O3 the "dominant" pollutant in 53 % of
   hours, and PM10 shows a step change from March 2025. So the headline AQI uses **PM2.5** by default (`AQI_BASIS=pm2_5`).
   The full six-pollutant NAQI is implemented, tested and selectable (`AQI_BASIS=all`).
3. **The model series has regime changes** (PM10/PM2.5 ratio shifts in Nov 2024 and Mar 2025, consistent with CAMS upgrades).
4. **The Diwali signal is weak and inconsistent** (1.2x to 2.1x the week before). Ground sensors usually show far sharper
   spikes; a coarse model grid cannot resolve local fireworks.
5. **Boundary-layer height is missing for Jan to Jun 2024** in the archive.

---

## Limitations (please read)

* **Model data, not sensors.** All headline values are CAMS model estimates on a ~40 km grid. They are not a substitute for
  official CPCB/DPCC readings, and the newest hours are model forecasts that get revised. **Not health advice.**
* **Sensor gaps.** The optional OpenAQ layer shows real sensors, which have gaps, faults and short histories; it is not used
  for training, and I could not verify it against the live API (see below).
* **Accuracy drops with horizon.** Past roughly 24 hours the forecast is only a modest improvement over persistence.
* **No stubble-burning satellite data.** Crop-residue fires (a major October/November driver) are represented only by a
  season flag. Real fire counts (NASA FIRMS) would be the first upgrade.
* **Short history.** Data starts in August 2022: at most four winters and four Diwali periods to learn from and test on.
* **AQI forecast band is approximate.** It is derived from the PM quantile paths (a quantile of a 24-hour average is not the
  average of quantiles), so the band is wider than ideal at long horizons.
* **Weather as a future input is not used.** Only past weather feeds the model, because historical weather is reanalysis
  while live weather is a forecast and mixing them would flatter the results.
* **Calendar flags are approximate.** Diwali dates are good to within a day; lunar holidays other than Diwali are not modelled.

---

## What was and was not tested

| Part | Status |
|---|---|
| Python pipeline, models, API, 130+ tests | Run and passing here |
| Real data download, training, evaluation, forecast | Run on real Open-Meteo data |
| Whole stack on **PostgreSQL** (ingest upserts, training, forecast, every endpoint) | Run here against a real PostgreSQL server; results match SQLite |
| Frontend: type-check, build, rendering at desktop/phone widths in light and dark | Run here in a real browser |
| `python run.py` from a clean checkout | Run here (see the log in the pull request) |
| `docker compose` file | Validated with `docker compose config`; **not run** (no Docker daemon where this was built) |
| OpenAQ ground stations | Written against the v3 docs, tested with mocked responses; **never run against the live API** |
| Live weather endpoint (`api.open-meteo.com`) | Blocked in the build sandbox; the code path is tested with mocks and degrades gracefully |
| Render / Vercel deployment, GitHub Actions workflows | **Not run** (they need your accounts) |

---

## Other ways to run it

### Docker Compose (PostgreSQL + separate worker)

```bash
docker compose up --build        # then open http://localhost:8000
```

Services: `db` (PostgreSQL 16), `init` (one-shot download + training), `api` (FastAPI + website), `worker` (hourly refresh,
weekly retrain). No `.env` is required.

### Development

```bash
python run.py --skip-bootstrap --no-browser     # backend on :8000 (creates .venv)
cd frontend && npm install && cp .env.example .env.local && npm run dev   # frontend on :3000
python scripts/build_frontend.py                # rebuild the site into backend/app/static (needs Node 20+)
make test                                       # offline tests (about 2 minutes)
make lint
```

### Commands

```bash
python -m pipeline.ingest backfill|update|status   # data
python -m pipeline.jobs bootstrap|refresh|train|forecast|worker
python -m ml.evaluate --write-seed                 # walk-forward evaluation (about 10 minutes)
python scripts/build_eda_notebook.py               # re-run the EDA notebook
```

---

## Configuration

Copy `.env.example` to `.env` to change anything. All settings are optional.

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | SQLite file in `data/` | Use `postgresql+psycopg://user:pass@host:5432/db` for PostgreSQL |
| `AQI_BASIS` | `pm2_5` | `pm2_5`, `pm` (PM2.5+PM10) or `all` (official six-pollutant NAQI) |
| `OPENAQ_API_KEY` | empty | Free key from explore.openaq.org: shows real ground sensors on the map |
| `ENABLE_SCHEDULER` | `true` | In-process hourly refresh (Docker Compose turns it off and uses the worker) |
| `REFRESH_MINUTES` / `RETRAIN_DAYS` | `60` / `7` | Refresh and retraining cadence |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Where the server listens |
| `CORS_ORIGINS` | `http://localhost:3000` | Only needed when the frontend runs on another origin |

---

## API

Interactive docs at `/docs` once running. All endpoints are read-only `GET`.

| Endpoint | Returns |
|---|---|
| `/api/health` | status, data freshness, model age |
| `/api/current?location=` | AQI, category, sub-indices, pollutants, weather, freshness |
| `/api/history?metric=&resolution=&start=&end=` | hourly / daily / monthly series |
| `/api/forecast?location=` | 72 h median and 80 % band for PM2.5, PM10 and AQI, plus the last 48 h observed |
| `/api/seasonal?metric=` | month × hour and year × month averages |
| `/api/stations` | map data: model cells, neighbourhoods, optional OpenAQ sensors |
| `/api/model-metrics` | the walk-forward evaluation |

---

## Deploy

**Status: written, not deployed or tested from this repository.** The Render/Neon free tiers change often, so check their
current limits first (at the time of writing, Render free web services sleep when idle and Render's free PostgreSQL expires
after a limited time, which is why a Neon free database is suggested).

1. **Database:** create a free PostgreSQL database (for example on Neon) and copy its connection string, rewritten to start
   with `postgresql+psycopg://`.
2. **Web service:** on Render choose *New → Blueprint*, point it at this repository (it reads `render.yaml`) and paste the
   database URL as `DATABASE_URL`. The container starts instantly, shows a "setup is running" message and bootstraps in the
   background (free instances have little memory; if it is killed, use step 3 instead).
3. **Keep it fresh and do the heavy first run on GitHub:** add the repository secret `DATABASE_URL`, run the workflow
   *Refresh data and forecast* once by hand with `command = bootstrap`, then set the repository variable
   `ENABLE_REFRESH=true` to turn on the hourly schedule.
4. Put the live link at the top of this README.

The site and API are one service, so Vercel is optional.

---

## Project layout

```
run.py                one-command launcher
pipeline/             config, db, ingest, aqi, features, openaq, jobs
ml/                   baselines, train, evaluate, forecast, registry (+ shipped metrics.json)
backend/app/          FastAPI: main, service (all logic), routes/, scheduler, static/ (built site)
frontend/             Next.js + TypeScript (static export)
notebooks/            01_eda.ipynb (executed)
scripts/              check_grid_cells.py, build_eda_notebook.py, build_frontend.py
tests/                pytest, offline
docs/ARCHITECTURE.md  the full design: data flow, every folder, why
```

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design and the reasoning behind it.

## Troubleshooting

* **`python` not found / too old:** install Python 3.11+ from python.org. On Windows try `py -3.11 run.py`.
* **First run seems stuck:** it is downloading and training; progress lines appear in the terminal. Open-Meteo may rate-limit
  heavy use, in which case the downloader waits and retries automatically.
* **Port already in use:** `python run.py --port 8080`.
* **Corporate proxy / no internet:** the app needs to reach `air-quality-api.open-meteo.com`, `archive-api.open-meteo.com` and
  `api.open-meteo.com`.
* **Start over:** delete `data/delhi_aqi.db` and run again.

## Credits and licence

Air-quality data: Copernicus Atmosphere Monitoring Service (CAMS) via [Open-Meteo](https://open-meteo.com/) (CC BY 4.0).
AQI breakpoints: CPCB National AQI (2014). Map tiles: © OpenStreetMap contributors. Code: MIT, see `LICENSE`.
