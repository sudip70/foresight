from __future__ import annotations

from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Query, Request

from backend.app.api.routes.dependencies import (
    _allow_artifact_fallback,
    _artifact_profile_payload,
    _market_forecast_engine,
    _market_index_cache_payload,
    _market_index_live_payload,
    _market_index_proxy_or_503,
    _require_engine,
)
from backend.app.api.schemas import (
    MarketIndexHistoryResponse,
    MarketIndexResponse,
    TickerProfileResponse,
    UniverseResponse,
)
from backend.app.core.config import get_settings
from backend.app.market.index_refresh import fetch_market_index_history_from_repository
from backend.app.market.repository import MarketDataUnavailable
from backend.app.ml.errors import ArtifactValidationError

router = APIRouter(tags=["market"])


@router.get("/universe", response_model=UniverseResponse)
def universe(request: Request):
    app = request.app
    try:
        forecast_engine = _market_forecast_engine(app)
        if forecast_engine is not None:
            try:
                return forecast_engine.universe_payload()
            except (ArtifactValidationError, MarketDataUnavailable):
                if not _allow_artifact_fallback():
                    raise
                pass
        if not _allow_artifact_fallback():
            raise HTTPException(status_code=503, detail="Supabase market data is not available")
        engine = _require_engine(app)
        return engine.universe_payload()
    except HTTPException:
        raise
    except ArtifactValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/tickers/{ticker}/profile", response_model=TickerProfileResponse)
def ticker_profile(ticker: str, request: Request):
    app = request.app
    try:
        forecast_engine = _market_forecast_engine(app)
        if forecast_engine is not None:
            try:
                return forecast_engine.ticker_profile_payload(ticker)
            except (ArtifactValidationError, MarketDataUnavailable):
                if not _allow_artifact_fallback():
                    raise
                pass
        if not _allow_artifact_fallback():
            raise HTTPException(status_code=503, detail="Supabase market data is not available")
        engine = _require_engine(app)
        return _artifact_profile_payload(engine, ticker)
    except HTTPException:
        raise
    except ArtifactValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/market/indices", response_model=MarketIndexResponse)
def market_indices(request: Request):
    app = request.app
    forecast_engine = _market_forecast_engine(app)
    if forecast_engine is None:
        cache_payload = _market_index_cache_payload(app)
        if cache_payload["indices"]:
            return cache_payload
        try:
            return _market_index_live_payload(app)
        except Exception as exc:  # pragma: no cover - provider/network defensive path
            app.state.market_index_refresh_error = exc
            return _market_index_proxy_or_503(app, exc)
    try:
        payload = forecast_engine.market_indices_payload()
        if payload.get("indices"):
            return payload
        cache_payload = _market_index_cache_payload(app)
        if cache_payload["indices"]:
            return cache_payload
        try:
            return _market_index_live_payload(app)
        except Exception as exc:  # pragma: no cover - provider/network defensive path
            app.state.market_index_refresh_error = exc
            return _market_index_proxy_or_503(app, exc)
    except MarketDataUnavailable:
        cache_payload = _market_index_cache_payload(app)
        if cache_payload["indices"]:
            return cache_payload
        try:
            return _market_index_live_payload(app)
        except Exception as live_exc:
            return _market_index_proxy_or_503(app, live_exc)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/market/indices/{symbol}/history",
    response_model=MarketIndexHistoryResponse,
)
def market_index_history(
    symbol: str,
    request: Request,
    history_range: str = Query(
        default="1y",
        alias="range",
        pattern="^(1m|3m|6m|1y|5y)$",
    ),
):
    from backend.app import main as app_main

    _HISTORY_CACHE_MAX = 50
    app = request.app
    settings = get_settings()
    normalized_symbol = symbol.strip().upper()
    normalized_range = history_range.strip().lower()
    cache_key = (normalized_symbol, normalized_range)
    history_cache: OrderedDict = getattr(app.state, "market_index_history_cache", OrderedDict())
    if not isinstance(history_cache, OrderedDict):
        history_cache = OrderedDict(history_cache)
    if cache_key in history_cache:
        history_cache.move_to_end(cache_key)
        return history_cache[cache_key]
    repository = getattr(app.state, "market_repository", None)
    repository_error: Exception | None = None
    if repository is not None:
        try:
            payload = fetch_market_index_history_from_repository(
                settings,
                repository=repository,
                symbol=normalized_symbol,
                history_range=normalized_range,
            )
            history_cache[cache_key] = payload
            while len(history_cache) > _HISTORY_CACHE_MAX:
                history_cache.popitem(last=False)
            app.state.market_index_history_cache = history_cache
            return payload
        except ValueError as repository_exc:
            if "Unsupported market index symbol" in str(repository_exc):
                raise HTTPException(
                    status_code=404,
                    detail=str(repository_exc),
                ) from repository_exc
            repository_error = repository_exc
        except Exception as repository_exc:
            repository_error = repository_exc
    try:
        payload = app_main.fetch_market_index_history(
            settings,
            symbol=normalized_symbol,
            history_range=normalized_range,
        )
        history_cache[cache_key] = payload
        while len(history_cache) > _HISTORY_CACHE_MAX:
            history_cache.popitem(last=False)
        app.state.market_index_history_cache = history_cache
        return payload
    except ValueError as exc:
        message = str(exc)
        if "Unsupported market index symbol" in message:
            raise HTTPException(status_code=404, detail=message) from exc
        if (
            "No index history" in message
            or "No usable index history" in message
            or "No close-price history" in message
        ):
            raise HTTPException(status_code=503, detail=message) from exc
        raise HTTPException(status_code=400, detail=message) from exc
    except ImportError:
        detail = (
            f"Market index history for {normalized_symbol} is not yet available. "
            "Run the data refresh pipeline to populate proxy ETF data."
        )
        raise HTTPException(status_code=503, detail=detail)
    except Exception as exc:
        detail = str(exc) or "Market index history is unavailable"
        if repository_error is not None and str(repository_error):
            detail = f"{detail}; Supabase proxy history unavailable: {repository_error}"
        raise HTTPException(status_code=503, detail=detail) from exc
