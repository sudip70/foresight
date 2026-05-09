from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.app.api.routes.dependencies import _local_refresh_status_payload, _require_engine
from backend.app.api.schemas import BacktestRequest, BacktestResponse, RefreshStatusResponse
from backend.app.core.config import get_settings
from backend.app.market.repository import MarketDataUnavailable
from backend.app.ml.errors import ArtifactValidationError

router = APIRouter(tags=["diagnostics"])


@router.get("/data/refresh/status", response_model=RefreshStatusResponse)
def refresh_status(request: Request):
    app = request.app
    repository = getattr(app.state, "market_repository", None)
    if repository is None:
        return _local_refresh_status_payload(app)
    try:
        return repository.get_latest_refresh_status()
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/backtests", response_model=BacktestResponse)
def backtests(request_body: BacktestRequest, request: Request):
    settings = get_settings()
    try:
        engine = _require_engine(request.app)
        result = engine.run_backtest(
            initial_amount=request_body.initial_amount,
            risk=request_body.risk,
            window_size=request_body.window_size,
            max_steps=request_body.max_steps or settings.default_backtest_steps,
            include_trade_log=request_body.include_trade_log,
        )
        return result.__dict__
    except HTTPException:
        raise
    except ArtifactValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
