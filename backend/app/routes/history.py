import datetime as dt

from fastapi import APIRouter, Depends, Query
from sqlalchemy.engine import Engine

from backend.app import service
from backend.app.deps import get_db

router = APIRouter(tags=["air quality"])


@router.get("/history", summary="Historical series (hourly, daily or monthly)")
def history(
    location: str = Query("delhi"),
    metric: str = Query("aqi", description="aqi, pm2_5, pm10, no2, so2, co or o3"),
    start: dt.datetime | None = Query(None, description="UTC, ISO 8601. Default: 90 days before the end"),
    end: dt.datetime | None = Query(None, description="UTC, ISO 8601. Default: latest data"),
    resolution: str = Query("daily", description="hourly (max 62 days), daily or monthly"),
    db: Engine = Depends(get_db),
):
    return service.history(db, location, metric, _naive(start), _naive(end), resolution)


@router.get("/seasonal", summary="Mean by month and hour of day (IST) for the seasonal heatmap")
def seasonal(location: str = Query("delhi"), metric: str = Query("aqi"), db: Engine = Depends(get_db)):
    return service.seasonal(db, location, metric)


def _naive(value: dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    return value.astimezone(dt.UTC).replace(tzinfo=None) if value.tzinfo else value
