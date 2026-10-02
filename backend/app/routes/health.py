from fastapi import APIRouter, Depends
from sqlalchemy.engine import Engine

from backend.app import service
from backend.app.deps import get_db

router = APIRouter(tags=["system"])


@router.get("/health", summary="Service status, data freshness and model age")
def health(db: Engine = Depends(get_db)):
    return service.health(db)


@router.get("/locations", summary="Locations the app tracks (model cells and the city average)")
def locations():
    return service.locations_list()
