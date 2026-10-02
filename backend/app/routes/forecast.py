from fastapi import APIRouter, Depends, Query
from sqlalchemy.engine import Engine

from backend.app import service
from backend.app.deps import get_db

router = APIRouter(tags=["forecast"])


@router.get("/forecast", summary="72-hour forecast with an 80% prediction interval")
def forecast(
    location: str = Query("delhi"),
    history_hours: int = Query(48, ge=0, le=240, description="Observed hours returned before the forecast starts"),
    db: Engine = Depends(get_db),
):
    return service.forecast(db, location, history_hours)
