from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.app.api.routes.dependencies import (
    _allow_artifact_fallback,
    _market_forecast_engine,
    _require_engine,
)
from backend.app.api.schemas import PortfolioSimulationRequest, PortfolioSimulationResponse
from backend.app.market.repository import MarketDataUnavailable
from backend.app.ml.errors import ArtifactValidationError

router = APIRouter(tags=["portfolio"])


@router.post(
    "/portfolio/simulations",
    response_model=PortfolioSimulationResponse,
)
def portfolio_simulation(request_body: PortfolioSimulationRequest, request: Request):
    app = request.app
    try:
        forecast_engine = _market_forecast_engine(app)
        if forecast_engine is not None:
            try:
                return forecast_engine.run_portfolio_simulation(
                    amount=request_body.amount,
                    risk=request_body.risk,
                    horizon_days=request_body.horizon_days,
                    selected_tickers=request_body.selected_tickers,
                    window_size=request_body.window_size,
                    max_crypto_weight=request_body.max_crypto_weight,
                    max_single_position_weight=request_body.max_single_position_weight,
                    min_cash_weight=request_body.min_cash_weight,
                    preferred_asset_classes=request_body.preferred_asset_classes,
                )
            except (ArtifactValidationError, MarketDataUnavailable):
                if not _allow_artifact_fallback():
                    raise
                pass
        if not _allow_artifact_fallback():
            raise HTTPException(status_code=503, detail="Supabase market data is not available")
        engine = _require_engine(app)
        result = engine.run_portfolio_simulation(
            amount=request_body.amount,
            risk=request_body.risk,
            horizon_days=request_body.horizon_days,
            selected_tickers=request_body.selected_tickers,
            window_size=request_body.window_size,
            max_crypto_weight=request_body.max_crypto_weight,
            max_single_position_weight=request_body.max_single_position_weight,
            min_cash_weight=request_body.min_cash_weight,
            preferred_asset_classes=request_body.preferred_asset_classes,
        )
        return result
    except HTTPException:
        raise
    except ArtifactValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
