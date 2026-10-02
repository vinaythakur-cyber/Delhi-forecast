from fastapi import APIRouter

from backend.app.routes import current, forecast, health, history, metrics, stations

api_router = APIRouter(prefix="/api")
for module in (health, current, history, forecast, stations, metrics):
    api_router.include_router(module.router)
