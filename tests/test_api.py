"""API tests: every route against a seeded database with a tiny trained model."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import backend.app.main as main_mod
import ml.registry as reg
import ml.train as train_mod
from backend.app import service
from backend.app.deps import get_db
from ml.forecast import make_forecast, train_final
from ml.registry import save_evaluation


@pytest.fixture()
def client(seeded_engine, monkeypatch, tmp_path):
    monkeypatch.setattr(train_mod, "N_ROUNDS", 20)
    monkeypatch.setattr(reg, "SEED_METRICS", tmp_path / "no-seed.json")
    service.clear_cache()
    app = main_mod.create_app(start_scheduler=False)
    app.dependency_overrides[get_db] = lambda: seeded_engine
    yield TestClient(app)
    service.clear_cache()


@pytest.fixture()
def ready_client(client, seeded_engine):
    train_final(seeded_engine, stride=12, n_random=1)
    make_forecast(seeded_engine)
    return client


def test_health_reports_initializing_until_a_model_exists(client, ready_client):
    body = ready_client.get("/api/health").json()
    assert body["status"] == "ok" and body["model_trained_at"] and body["aqi_basis"] == "pm2_5"
    assert body["data_through"].endswith("Z")


def test_health_initializing_without_model(client):
    assert client.get("/api/health").json()["status"] == "initializing"


@pytest.mark.parametrize("location", ["delhi", "delhi-north", "delhi-south"])
def test_current_has_a_consistent_aqi_card(ready_client, location):
    body = ready_client.get("/api/current", params={"location": location}).json()
    aqi = body["aqi"]
    assert 0 <= aqi["value"] <= 500 and aqi["category"] and aqi["color"].startswith("#")
    assert aqi["basis"] == "pm2_5" and aqi["dominant"] == "pm2_5"
    assert [p["key"] for p in body["pollutants"]] == ["pm2_5", "pm10", "no2", "so2", "co", "o3"]
    assert [p["key"] for p in body["pollutants"] if p["counts_toward_aqi"]] == ["pm2_5"]
    assert all(p["sub_index"] is not None for p in body["pollutants"])
    assert body["freshness"]["data_through"] == body["observed_at"] or body["freshness"]["data_through"]
    assert [c["name"] for c in body["categories"]][0] == "Good" and len(body["categories"]) == 6
    assert set(body["weather"]) >= {"temperature_c", "wind_kmh", "boundary_layer_m"}


def test_unknown_location_is_404_and_empty_database_is_503(ready_client, tmp_path):
    from pipeline.db import get_engine, init_db

    assert ready_client.get("/api/current", params={"location": "mars"}).status_code == 404
    empty = init_db(get_engine(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"))
    app = main_mod.create_app(start_scheduler=False)
    app.dependency_overrides[get_db] = lambda: empty
    res = TestClient(app).get("/api/current")
    assert res.status_code == 503 and res.json()["status"] == "initializing"


def test_history_resolutions_and_validation(ready_client):
    daily = ready_client.get("/api/history").json()
    assert daily["resolution"] == "daily" and 60 <= len(daily["points"]) <= 91
    p = daily["points"][0]
    assert len(p["t"]) == 10 and p["min"] <= p["v"] <= p["max"]
    monthly = ready_client.get("/api/history", params={"resolution": "monthly", "metric": "pm2_5", "start": "2025-01-01T00:00:00Z"}).json()
    assert 5 <= len(monthly["points"]) <= 14 and monthly["unit"] == "µg/m³"
    hourly = ready_client.get("/api/history", params={"resolution": "hourly", "start": "2025-12-10T00:00:00Z"}).json()
    assert len(hourly["points"]) > 300 and hourly["points"][0]["t"].endswith("Z")
    assert ready_client.get("/api/history", params={"resolution": "hourly", "start": "2025-01-01T00:00:00Z"}).status_code == 422
    assert ready_client.get("/api/history", params={"metric": "nonsense"}).status_code == 422
    assert ready_client.get("/api/history", params={"resolution": "yearly"}).status_code == 422


def test_forecast_shape_ordering_and_context(ready_client):
    body = ready_client.get("/api/forecast").json()
    pts = body["points"]
    assert [p["horizon"] for p in pts] == list(range(1, 73))
    for p in pts:
        for t in ("pm2_5", "pm10", "aqi"):
            assert p[t]["q10"] <= p[t]["q50"] <= p[t]["q90"]
        assert p["category"] and p["color"]
    assert len(body["recent"]) == 48 and body["recent"][-1]["t"] == body["issued_at"]
    assert pts[0]["target_ts"] > body["issued_at"]
    assert "80%" in body["interval"] and body["caveat"]
    short = ready_client.get("/api/forecast", params={"history_hours": 6}).json()
    assert len(short["recent"]) == 6
    assert ready_client.get("/api/forecast", params={"history_hours": 9999}).status_code == 422


def test_forecast_503_before_the_first_forecast(client):
    assert client.get("/api/forecast").status_code == 503


def test_stations_returns_cells_neighbourhoods_and_no_ground_data_without_a_key(ready_client):
    body = ready_client.get("/api/stations").json()
    assert {c["id"] for c in body["cells"]} == {"delhi-north", "delhi-south"}
    assert len(body["neighbourhoods"]) == 21
    cell_aqi = {c["id"]: c["aqi"] for c in body["cells"]}
    assert all(n["aqi"] == cell_aqi[n["cell"]] for n in body["neighbourhoods"])  # members share their cell's value
    assert body["ground_stations"] == [] and body["ground_stations_enabled"] is False and "coarse" in body["note"]


def test_seasonal_heatmap_grid(ready_client):
    body = ready_client.get("/api/seasonal", params={"metric": "pm2_5"}).json()
    assert len(body["months"]) == 12 and len(body["month_by_hour"]) == 12 and all(len(r) == 24 for r in body["month_by_hour"])
    assert body["year_month"] and body["timezone"].startswith("IST")


def test_model_metrics_requires_an_evaluation_then_serves_it(ready_client, seeded_engine):
    assert ready_client.get("/api/model-metrics").status_code == 503
    save_evaluation(seeded_engine, {"generated_at": "now", "targets": {}})
    body = ready_client.get("/api/model-metrics").json()
    assert body["evaluation"]["generated_at"] == "now" and body["production_model"]["trained_at"]
    assert any("two independent cells" in s for s in body["limitations"])


def test_locations_endpoint_lists_city_and_cells(client):
    ids = [loc["id"] for loc in client.get("/api/locations").json()]
    assert ids == ["delhi", "delhi-north", "delhi-south"]


def test_cors_allows_the_dev_frontend_origin(client):
    res = client.get("/api/locations", headers={"Origin": "http://localhost:3000"})
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_frontend_is_served_at_root_when_built(seeded_engine, monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<html><body>Delhi AQI</body></html>")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "index.html").write_text("<html>model page</html>")
    monkeypatch.setattr(main_mod, "STATIC_DIR", tmp_path)
    c = TestClient(main_mod.create_app(start_scheduler=False))
    assert "Delhi AQI" in c.get("/").text and "model page" in c.get("/model/").text
    assert c.get("/api/locations").status_code == 200  # the API still wins over the static mount


def test_api_docs_are_available(client):
    schema = client.get("/openapi.json").json()
    assert {"/api/current", "/api/history", "/api/forecast", "/api/stations", "/api/model-metrics", "/api/seasonal"} <= set(schema["paths"])
