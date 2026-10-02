from fastapi import APIRouter, Depends, Query
from sqlalchemy.engine import Engine

from backend.app import service
from backend.app.deps import get_db

router = APIRouter(tags=["air quality"])


@router.get("/current", summary="Current AQI, pollutants, weather and data freshness")
def current(location: str = Query("delhi", description="delhi, delhi-north or delhi-south"), db: Engine = Depends(get_db)):
    return service.current(db, location)
