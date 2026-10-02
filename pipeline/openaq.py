"""Optional real ground stations from OpenAQ v3. Off unless OPENAQ_API_KEY is set.

STATUS: written against OpenAQ's published v3 response shape and covered by mocked-response tests,
but NOT verified against the live API (the key is yours to add). Every parse step is defensive:
anything unexpected is skipped, so a schema surprise degrades to "no ground stations", never to an
error on the site. Results are cached in memory for 15 minutes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from pipeline.aqi import category_for, sub_index
from pipeline.config import Settings, get_settings

log = logging.getLogger("pipeline.openaq")
CACHE_SECONDS = 15 * 60
_cache: dict[str, Any] = {"at": 0.0, "data": []}
DELHI_CENTRE = "28.6139,77.2090"
RADIUS_M = 25000  # OpenAQ's maximum search radius


def _pm25_sensor_ids(location: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for sensor in location.get("sensors") or []:
        param = (sensor.get("parameter") or {}).get("name", "")
        if str(param).lower() in {"pm25", "pm2.5", "pm2_5"} and "id" in sensor:
            ids.add(int(sensor["id"]))
    return ids


def parse_stations(locations_payload: dict[str, Any], latest_by_location: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge the location list with each location's latest PM2.5 reading."""
    out: list[dict[str, Any]] = []
    for loc in locations_payload.get("results") or []:
        try:
            loc_id = int(loc["id"])
            coords = loc.get("coordinates") or {}
            lat, lon = float(coords["latitude"]), float(coords["longitude"])
            sensors = _pm25_sensor_ids(loc)
            if not sensors:
                continue
            reading = None
            for item in (latest_by_location.get(loc_id) or {}).get("results") or []:
                if int(item.get("sensorsId", -1)) in sensors and item.get("value") is not None:
                    reading = item
                    break
            if reading is None:
                continue
            value = float(reading["value"])
            if not 0 <= value <= 1500:
                continue
            si = float(sub_index("pm2_5", value))
            cat = category_for(si)
            out.append(
                {
                    "id": loc_id,
                    "name": str(loc.get("name") or f"Station {loc_id}"),
                    "lat": lat,
                    "lon": lon,
                    "pm2_5": round(value, 1),
                    "pm2_5_sub_index": round(si),
                    "category": cat.name if cat else None,
                    "color": cat.color if cat else None,
                    "observed_at": ((reading.get("datetime") or {}).get("utc")),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def fetch_ground_stations(settings: Settings | None = None, client: httpx.Client | None = None, use_cache: bool = True) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    if not settings.openaq_api_key:
        return []
    if use_cache and time.time() - _cache["at"] < CACHE_SECONDS:
        return _cache["data"]
    own = client is None
    client = client or httpx.Client(timeout=30)
    headers = {"X-API-Key": settings.openaq_api_key}
    try:
        resp = client.get(
            f"{settings.openaq_base_url}/locations",
            params={"coordinates": DELHI_CENTRE, "radius": RADIUS_M, "limit": 200},
            headers=headers,
        )
        resp.raise_for_status()
        payload = resp.json()
        latest: dict[int, dict[str, Any]] = {}
        for loc in (payload.get("results") or [])[:60]:  # bound the number of follow-up calls
            try:
                loc_id = int(loc["id"])
                r = client.get(f"{settings.openaq_base_url}/locations/{loc_id}/latest", headers=headers)
                if r.status_code == 200:
                    latest[loc_id] = r.json()
            except (KeyError, TypeError, ValueError, httpx.HTTPError):
                continue
        data = parse_stations(payload, latest)
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("OpenAQ unavailable (%s); showing model cells only", exc)
        data = []
    finally:
        if own:
            client.close()
    _cache.update(at=time.time(), data=data)
    return data
