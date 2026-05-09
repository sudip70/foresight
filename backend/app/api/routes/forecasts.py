from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.app.api.routes.dependencies import (
    _allow_artifact_fallback,
    _market_forecast_engine,
    _require_engine,
)
from backend.app.api.schemas import (
    MarketForecastRequest,
    MarketForecastResponse,
    TickerForecastRequest,
    TickerForecastResponse,
)
from backend.app.market.repository import MarketDataUnavailable
from backend.app.ml.errors import ArtifactValidationError

router = APIRouter(tags=["forecasts"])


@router.post("/forecasts/ticker", response_model=TickerForecastResponse)
def ticker_forecast(request_body: TickerForecastRequest, request: Request):
    app = request.app
    try:
        forecast_engine = _market_forecast_engine(app)
        if forecast_engine is not None:
            try:
                return forecast_engine.run_ticker_forecast(
                    ticker=request_body.ticker,
                    horizon_days=request_body.horizon_days,
                    window_size=request_body.window_size,
                )
            except (ArtifactValidationError, MarketDataUnavailable):
                if not _allow_artifact_fallback():
                    raise
                pass
        if not _allow_artifact_fallback():
            raise HTTPException(status_code=503, detail="Supabase market data is not available")
        engine = _require_engine(app)
        return engine.run_ticker_forecast(
            ticker=request_body.ticker,
            horizon_days=request_body.horizon_days,
            window_size=request_body.window_size,
        )
    except HTTPException:
        raise
    except ArtifactValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/forecasts/market", response_model=MarketForecastResponse)
def market_forecast(request_body: MarketForecastRequest, request: Request):
    app = request.app
    try:
        forecast_engine = _market_forecast_engine(app)
        if forecast_engine is not None:
            try:
                return forecast_engine.run_market_forecast(
                    horizon_days=request_body.horizon_days,
                    risk=request_body.risk,
                    top_n=request_body.top_n,
                    window_size=request_body.window_size,
                )
            except (ArtifactValidationError, MarketDataUnavailable):
                if not _allow_artifact_fallback():
                    raise
                pass
        if not _allow_artifact_fallback():
            raise HTTPException(status_code=503, detail="Supabase market data is not available")
        engine = _require_engine(app)
        return engine.run_market_forecast(
            horizon_days=request_body.horizon_days,
            risk=request_body.risk,
            top_n=request_body.top_n,
            window_size=request_body.window_size,
        )
    except HTTPException:
        raise
    except ArtifactValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
