from fastapi import APIRouter, Depends
from sqlalchemy.engine import Engine

from backend.app import service
from backend.app.deps import get_db
from pipeline import openaq

router = APIRouter(tags=["map"])


@router.get("/stations", summary="Map data: model cells, neighbourhoods and (optional) OpenAQ ground stations")
def stations(db: Engine = Depends(get_db)):
    return service.stations(db, openaq.fetch_ground_stations())
