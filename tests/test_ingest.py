"""Ingestion tests with a mocked Open-Meteo (no network)."""

from __future__ import annotations

import datetime as dt

import httpx
import numpy as np
import pandas as pd
import pytest

from pipeline import ingest
from pipeline.config import get_settings
from pipeline.db import count_rows, observations, read_observations, upsert_observations
from pipeline.locations import CELLS


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(ingest, "_sleep", lambda s: None)


NOW = dt.datetime(2026, 3, 10, 12, 30)


def _payload(start: str, hours: int, base: float = 50.0, mapping=ingest.AQ_VARS) -> dict:
    times = pd.date_range(start, periods=hours, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    hourly = {"time": times}
    for api_name in mapping.values():
        hourly[api_name] = [base + (i % 24) for i in range(hours)]
    return {"hourly": hourly}


def make_handler(calls: list[str], weather_ok: bool = True):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        params = dict(request.url.params)
        start = params.get("start_date", "2026-03-01")
        if "air-quality" in url:
            return httpx.Response(200, json=_payload(start + "T00:00", 24 * 3))
        if "archive" in url:
            return httpx.Response(200, json=_payload(start + "T00:00", 24 * 3, mapping=ingest.WX_VARS))
        if not weather_ok:
            return httpx.Response(403, json={"reason": "blocked"})
        return httpx.Response(200, json=_payload("2026-03-08T00:00", 24 * 3, mapping=ingest.WX_VARS))

    return handler


def test_parse_hourly_maps_names_and_coerces_nulls():
    payload = {"hourly": {"time": ["2026-01-01T00:00", "2026-01-01T01:00"], "pm2_5": [10, None], "pm10": ["20", 30]}}
    df = ingest.parse_hourly(payload, {"pm2_5": "pm2_5", "pm10": "pm10", "no2": "nitrogen_dioxide"})
    assert list(df.columns) == ["pm2_5", "pm10", "no2"]
    assert df["pm2_5"].tolist()[0] == 10 and np.isnan(df["pm2_5"].tolist()[1])
    assert df["pm10"].tolist() == [20, 30]
    assert df["no2"].isna().all()  # variable missing from the payload becomes NaN, not an error
    assert df.index.name == "ts" and df.index.tz is None


def test_parse_hourly_empty_payload():
    assert ingest.parse_hourly({}, ingest.AQ_VARS).empty


def test_validate_removes_duplicates_and_impossible_values():
    idx = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 01:00", "2026-01-01 02:00"])
    df = pd.DataFrame({"pm2_5": [10.0, 20.0, 25.0, -5.0], "rh": [50.0, 120.0, 55.0, 40.0]}, index=idx)
    clean, report = ingest.validate_observations(df)
    assert len(clean) == 3 and report["duplicates"] == 1
    assert report["out_of_range"] == {"pm2_5": 1}  # the negative concentration
    assert np.isnan(clean.loc["2026-01-01 02:00", "pm2_5"])
    assert clean.loc["2026-01-01 01:00", "pm2_5"] == 25.0  # last duplicate wins
    assert clean.loc["2026-01-01 01:00", "rh"] == 55.0


def test_chunks_cover_range_without_overlap():
    chunks = list(ingest._chunks(dt.date(2022, 8, 4), dt.date(2026, 10, 2)))
    assert chunks[0][0] == dt.date(2022, 8, 4) and chunks[-1][1] == dt.date(2026, 10, 2)
    for (_, prev_end), (nxt_start, _) in zip(chunks, chunks[1:], strict=False):
        assert nxt_start == prev_end + dt.timedelta(days=1)
    assert all((b - a).days < 366 for a, b in chunks)


def test_get_json_retries_then_succeeds():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(429 if attempts["n"] < 3 else 200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert ingest.get_json(client, "https://x", {}, get_settings()) == {"ok": True}
    assert attempts["n"] == 3


def test_get_json_raises_on_client_error_with_reason():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400, json={"reason": "bad coords"})))
    with pytest.raises(ingest.SourceError, match="bad coords"):
        ingest.get_json(client, "https://x", {}, get_settings())


def test_ingest_location_is_idempotent_and_never_stores_future_hours(engine):
    calls: list[str] = []
    client = httpx.Client(transport=httpx.MockTransport(make_handler(calls)))
    start = dt.date(2026, 3, 8)
    n1, report = ingest.ingest_location(engine, client, CELLS[0], start, dt.date(2026, 3, 10), now=NOW)
    rows_after_first = count_rows(engine, observations)
    n2, _ = ingest.ingest_location(engine, client, CELLS[0], start, dt.date(2026, 3, 10), now=NOW)
    assert count_rows(engine, observations) == rows_after_first  # re-running adds nothing
    assert n1 == n2 > 0
    df = read_observations(engine, CELLS[0].id)
    assert df.index.max() <= pd.Timestamp("2026-03-10 12:00")  # nothing after the current hour
    assert report["null_fraction"]["pm2_5"] == 0


def test_ingest_degrades_when_recent_weather_is_unavailable(engine):
    calls: list[str] = []
    client = httpx.Client(transport=httpx.MockTransport(make_handler(calls, weather_ok=False)))
    state: dict[str, bool] = {}
    ingest.ingest_location(engine, client, CELLS[0], dt.date(2026, 3, 8), now=NOW, state=state)
    assert state["recent_weather_ok"] is False
    n_forecast_calls = sum(c.startswith("https://api.open-meteo.com") for c in calls)
    ingest.ingest_location(engine, client, CELLS[1], dt.date(2026, 3, 8), now=NOW, state=state)
    assert sum(c.startswith("https://api.open-meteo.com") for c in calls) == n_forecast_calls  # not retried per location
    assert not read_observations(engine, CELLS[1].id).empty


def test_build_city_is_mean_of_cells(engine):
    idx = pd.date_range("2026-01-01", periods=3, freq="h").as_unit("ns")
    upsert_observations(engine, CELLS[0].id, pd.DataFrame({"pm2_5": [10.0, 20, 30], "wind_dir": [350.0, 350, 350]}, index=idx))
    upsert_observations(engine, CELLS[1].id, pd.DataFrame({"pm2_5": [30.0, 40, 50], "wind_dir": [10.0, 10, 10]}, index=idx))
    ingest.build_city(engine)
    city = read_observations(engine, "delhi")
    assert city["pm2_5"].tolist() == [20.0, 30.0, 40.0]
    assert city["wind_dir"].iloc[0] == pytest.approx(0.0, abs=1e-6)  # circular mean of 350 and 10 degrees
