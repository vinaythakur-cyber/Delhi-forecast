# data/

This folder is where the app keeps its local SQLite database (`delhi_aqi.db`) and any raw
downloads. Everything in it except this file is git-ignored: data is **fetched, never committed**.

Fetch it with one command from the repo root:

```bash
python -m pipeline.jobs bootstrap      # backfill history + train + forecast
python -m pipeline.ingest backfill     # only download history
python -m pipeline.ingest update       # only top up the latest hours
```

## Sources and which one is the source of truth

| Source | Role | Key? | Notes |
|---|---|---|---|
| **Open-Meteo Air Quality API** (CAMS model) | **Source of truth for the series the model learns and forecasts** | none | Gap-free hourly PM2.5, PM10, NO2, SO2, CO, O3 from 2022-08-04. It is a *model* on a coarse grid, not a sensor. |
| **Open-Meteo Weather archive / forecast API** (ERA5) | Weather covariates (boundary-layer height, wind, humidity, ...) | none | Archive lags ~5 days, so recent days come from the forecast endpoint. |
| **OpenAQ v3** | Optional real ground stations on the map | free key | Off unless `OPENAQ_API_KEY` is set. Gaps and sensor faults are expected. |
| Kaggle / CPCB downloads | Not used by the app | token / manual | Useful for offline validation of the model series against ground sensors. |

Open-Meteo is free for non-commercial use under CC BY 4.0 (attribution is shown in the site
footer). Check the current terms before using it commercially.
