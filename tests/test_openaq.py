"""OpenAQ is optional and unverified against the live API; these tests pin down the defensive parsing."""

from __future__ import annotations

import httpx
import pytest

from pipeline import openaq
from pipeline.config import Settings

LOCATIONS = {
    "results": [
        {"id": 1, "name": "Anand Vihar, Delhi - DPCC", "coordinates": {"latitude": 28.65, "longitude": 77.31},
         "sensors": [{"id": 11, "parameter": {"name": "pm25"}}, {"id": 12, "parameter": {"name": "no2"}}]},
        {"id": 2, "name": "No PM sensor", "coordinates": {"latitude": 28.6, "longitude": 77.2}, "sensors": [{"id": 21, "parameter": {"name": "o3"}}]},
        {"id": 3, "name": "Broken", "coordinates": {}, "sensors": [{"id": 31, "parameter": {"name": "pm25"}}]},
        {"id": 4, "name": "Absurd value", "coordinates": {"latitude": 28.7, "longitude": 77.1}, "sensors": [{"id": 41, "parameter": {"name": "pm25"}}]},
    ]
}
LATEST = {
    1: {"results": [{"sensorsId": 12, "value": 40}, {"sensorsId": 11, "value": 135.5, "datetime": {"utc": "2026-10-02T12:00:00Z"}}]},
    2: {"results": [{"sensorsId": 21, "value": 80}]},
    4: {"results": [{"sensorsId": 41, "value": 99999}]},
}


def test_parse_keeps_only_valid_pm25_stations_and_computes_a_sub_index():
    out = openaq.parse_stations(LOCATIONS, LATEST)
    assert [s["id"] for s in out] == [1]
    s = out[0]
    assert s["pm2_5"] == 135.5 and s["observed_at"] == "2026-10-02T12:00:00Z"
    assert 300 < s["pm2_5_sub_index"] <= 320 and s["category"] == "Very Poor" and s["color"].startswith("#")


def test_parse_survives_garbage():
    assert openaq.parse_stations({}, {}) == []
    assert openaq.parse_stations({"results": [None, 5, {"id": "x"}]}, {}) == []


def test_disabled_without_a_key():
    assert openaq.fetch_ground_stations(Settings(openaq_api_key="")) == []


def test_fetch_uses_the_key_header_and_degrades_gracefully():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-api-key")
        if request.url.path.endswith("/locations"):
            return httpx.Response(200, json=LOCATIONS)
        loc_id = int(request.url.path.split("/")[-2])
        return httpx.Response(200, json=LATEST.get(loc_id, {"results": []}))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    got = openaq.fetch_ground_stations(Settings(openaq_api_key="secret"), client, use_cache=False)
    assert seen["key"] == "secret" and [s["id"] for s in got] == [1]

    failing = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"message": "bad key"})))
    assert openaq.fetch_ground_stations(Settings(openaq_api_key="wrong"), failing, use_cache=False) == []


@pytest.fixture(autouse=True)
def reset_cache():
    openaq._cache.update(at=0.0, data=[])
    yield
    openaq._cache.update(at=0.0, data=[])
