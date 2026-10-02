from fastapi import APIRouter, Depends
from sqlalchemy.engine import Engine

from backend.app import service
from backend.app.deps import get_db

router = APIRouter(tags=["model"])


@router.get("/model-metrics", summary="Walk-forward evaluation: MAE/RMSE per horizon versus baselines")
def model_metrics(db: Engine = Depends(get_db)):
    return service.model_metrics(db)
