from __future__ import annotations

from backend.app.api.routes.diagnostics import router as diagnostics_router
from backend.app.api.routes.forecasts import router as forecasts_router
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.inference import router as inference_router
from backend.app.api.routes.market import router as market_router
from backend.app.api.routes.portfolio import router as portfolio_router

api_routers = (
    health_router,
    market_router,
    forecasts_router,
    portfolio_router,
    inference_router,
    diagnostics_router,
)
