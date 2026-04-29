from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    ExplanationRequest,
    ExplanationResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    MarketForecastRequest,
    MarketForecastResponse,
    MarketIndexResponse,
    ModelsResponse,
    PortfolioSimulationRequest,
    PortfolioSimulationResponse,
    RefreshStatusResponse,
    TickerProfileResponse,
    TickerForecastRequest,
    TickerForecastResponse,
    UniverseResponse,
)
from backend.app.core.config import REPO_ROOT, get_settings
from backend.app.market.forecasting import SupabaseForecastEngine
from backend.app.market.index_refresh import refresh_market_index_snapshots
from backend.app.market.repository import (
    MarketDataUnavailable,
    build_market_repository,
    empty_refresh_status,
)
from backend.app.ml.artifacts import ArtifactValidationError
from backend.app.ml.explainability import ExplainabilityUnavailable, build_explanations
from backend.app.ml.pipeline import get_engine


def _load_engine():
    return get_engine(get_settings(), strict_validation=True)


def _load_market_repository():
    return build_market_repository(get_settings())


def _market_health_payload(app: FastAPI) -> dict | None:
    repository = getattr(app.state, "market_repository", None)
    if repository is None:
        return None
    return repository.health_payload()


def _degraded_health_payload(app: FastAPI) -> dict:
    settings = get_settings()
    error = getattr(app.state, "engine_error", None)
    market_data = _market_health_payload(app)
    market_ready = bool(market_data and market_data.get("status") == "ok")
    return {
        "status": "ok" if market_ready else "degraded",
        "artifact_root": str(settings.artifact_root),
        "dataset_root": str(settings.dataset_root),
        "combined_aligned_rows": 0,
        "combined_date_start": "",
        "combined_date_end": "",
        "dependencies": {},
        "artifacts": {},
        "meta_model": {},
        "ready": market_ready,
        "error": str(error) if error else "Foresight engine is not ready",
        "market_data": market_data,
    }


def _require_engine(app: FastAPI):
    engine = getattr(app.state, "engine", None)
    if engine is None:
        error = getattr(app.state, "engine_error", None)
        detail = "Foresight engine is not ready"
        if error is not None:
            detail = f"{detail}: {error}"
        raise HTTPException(status_code=503, detail=detail)
    return engine


def _market_forecast_engine(app: FastAPI):
    return getattr(app.state, "forecast_engine", None)


def _market_index_cache_payload(app: FastAPI) -> dict:
    rows = getattr(app.state, "market_index_rows", []) or []
    refresh_result = getattr(app.state, "market_index_refresh_result", None) or {}
    error = getattr(app.state, "market_index_refresh_error", None)
    as_of_dates = [str(row["as_of_date"]) for row in rows if row.get("as_of_date")]
    disclaimer = "Index levels are refreshed from the configured market data provider at backend startup."
    if error is not None:
        disclaimer = f"{disclaimer} Latest startup refresh failed: {error}"
    return {
        "source": refresh_result.get("provider") or "startup_refresh",
        "as_of_date": refresh_result.get("as_of_date") or (max(as_of_dates) if as_of_dates else None),
        "indices": sorted(rows, key=lambda row: int(row.get("display_order") or 0)),
        "disclaimer": disclaimer,
    }


def _local_refresh_status_payload(app: FastAPI) -> dict:
    payload = empty_refresh_status()
    payload["message"] = "Using local artifact market data; Supabase refresh logs are not configured."

    engine = getattr(app.state, "engine", None)
    if engine is not None:
        try:
            universe = engine.universe_payload()
            tickers = universe.get("tickers") or []
            payload["latest_market_date"] = universe.get("latest_date") or None
            payload["asset_count"] = len(tickers)
        except Exception as exc:  # pragma: no cover - defensive diagnostics path
            payload["message"] = f"{payload['message']} Local artifact freshness is unavailable: {exc}"

    refresh_result = getattr(app.state, "market_index_refresh_result", None)
    refresh_error = getattr(app.state, "market_index_refresh_error", None)
    if refresh_result:
        payload["market_index_refresh"] = {
            "status": "ok",
            "provider": refresh_result.get("provider"),
            "as_of_date": refresh_result.get("as_of_date"),
            "rows": len(refresh_result.get("rows", []) or []),
            "rows_written": refresh_result.get("rows_written"),
        }
    elif refresh_error is not None:
        payload["market_index_refresh"] = {
            "status": "failed",
            "message": str(refresh_error),
        }

    return payload


@lru_cache(maxsize=1)
def _asset_universe_metadata() -> dict[str, dict]:
    path = REPO_ROOT / "config" / "asset_universe.v1.json"
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    assets = payload.get("assets", [])
    if not isinstance(assets, list):
        return {}
    return {str(row.get("ticker", "")).upper(): row for row in assets if row.get("ticker")}


def _artifact_profile_payload(engine, ticker: str) -> dict:
    asset_class, index, canonical = engine._resolve_ticker(ticker)
    context = engine._combined_context["assets"][asset_class]
    latest_index = engine._combined_context["aligned_rows"] - 1
    latest_date = str(engine._combined_context["dates"][latest_index])
    price = float(context["risky_prices"][latest_index, index])
    ohlcv = context["ohlcv"][latest_index, index]
    lookback_start = max(0, latest_index - 251)
    ohlcv_window = context["ohlcv"][lookback_start : latest_index + 1, index]
    metadata = _asset_universe_metadata().get(canonical.upper(), {})
    return {
        "ticker": canonical,
        "asset_class": asset_class,
        "display_name": metadata.get("display_name") or canonical,
        "as_of_date": None,
        "data_as_of": latest_date,
        "source": "local_artifacts",
        "fields": {
            "bid": None,
            "ask": None,
            "last_sale": price,
            "open": float(ohlcv[0]),
            "high": float(ohlcv[1]),
            "low": float(ohlcv[2]),
            "exchange": metadata.get("exchange"),
            "market_cap": None,
            "pe_ratio": None,
            "fifty_two_week_high": float(ohlcv_window[:, 1].max()),
            "fifty_two_week_low": float(ohlcv_window[:, 2].min()),
            "volume": float(ohlcv[4]),
            "average_volume": None,
            "margin_requirement": None,
            "dividend_frequency": None,
            "dividend_yield": None,
            "ex_dividend_date": None,
        },
    }


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = None
        app.state.engine_error = None
        app.state.market_repository = None
        app.state.market_repository_error = None
        app.state.forecast_engine = None
        app.state.market_index_rows = []
        app.state.market_index_refresh_result = None
        app.state.market_index_refresh_error = None
        try:
            app.state.market_repository = _load_market_repository()
            if app.state.market_repository is not None:
                app.state.forecast_engine = SupabaseForecastEngine(
                    app.state.market_repository,
                    settings,
                )
        except Exception as exc:  # pragma: no cover - defensive startup path
            app.state.market_repository_error = exc
        should_refresh_indices = settings.market_index_auto_refresh and (
            app.state.market_repository is None
            or bool(settings.supabase_url and settings.supabase_service_role_key)
        )
        if should_refresh_indices:
            try:
                refresh_result = refresh_market_index_snapshots(
                    settings,
                    repository=app.state.market_repository,
                )
                app.state.market_index_refresh_result = refresh_result
                app.state.market_index_rows = refresh_result.get("rows", [])
            except Exception as exc:  # pragma: no cover - provider/network defensive path
                app.state.market_index_refresh_error = exc
        try:
            app.state.engine = _load_engine()
        except Exception as exc:  # pragma: no cover - defensive startup path
            app.state.engine_error = exc
        yield

    app = FastAPI(
        title=settings.project_name,
        version="1.0.0",
        description="Ticker intelligence, scenario forecasting, and portfolio simulation API for Foresight.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get(f"{settings.api_prefix}/health", response_model=HealthResponse)
    def health():
        engine = getattr(app.state, "engine", None)
        if engine is None:
            return _degraded_health_payload(app)
        payload = engine.health_payload()
        payload["ready"] = True
        payload["market_data"] = _market_health_payload(app)
        return payload

    @app.get(f"{settings.api_prefix}/models", response_model=ModelsResponse)
    def models():
        try:
            engine = _require_engine(app)
            return engine.model_payload()
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get(f"{settings.api_prefix}/universe", response_model=UniverseResponse)
    def universe():
        try:
            forecast_engine = _market_forecast_engine(app)
            if forecast_engine is not None:
                try:
                    return forecast_engine.universe_payload()
                except (ArtifactValidationError, MarketDataUnavailable):
                    pass
            engine = _require_engine(app)
            return engine.universe_payload()
        except HTTPException:
            raise
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get(f"{settings.api_prefix}/tickers/{{ticker}}/profile", response_model=TickerProfileResponse)
    def ticker_profile(ticker: str):
        try:
            forecast_engine = _market_forecast_engine(app)
            if forecast_engine is not None:
                try:
                    return forecast_engine.ticker_profile_payload(ticker)
                except (ArtifactValidationError, MarketDataUnavailable):
                    pass
            engine = _require_engine(app)
            return _artifact_profile_payload(engine, ticker)
        except HTTPException:
            raise
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(f"{settings.api_prefix}/forecasts/ticker", response_model=TickerForecastResponse)
    def ticker_forecast(request: TickerForecastRequest):
        try:
            forecast_engine = _market_forecast_engine(app)
            if forecast_engine is not None:
                try:
                    return forecast_engine.run_ticker_forecast(
                        ticker=request.ticker,
                        horizon_days=request.horizon_days,
                        window_size=request.window_size,
                    )
                except (ArtifactValidationError, MarketDataUnavailable):
                    pass
            engine = _require_engine(app)
            return engine.run_ticker_forecast(
                ticker=request.ticker,
                horizon_days=request.horizon_days,
                window_size=request.window_size,
            )
        except HTTPException:
            raise
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(f"{settings.api_prefix}/forecasts/market", response_model=MarketForecastResponse)
    def market_forecast(request: MarketForecastRequest):
        try:
            forecast_engine = _market_forecast_engine(app)
            if forecast_engine is not None:
                try:
                    return forecast_engine.run_market_forecast(
                        horizon_days=request.horizon_days,
                        risk=request.risk,
                        top_n=request.top_n,
                        window_size=request.window_size,
                    )
                except (ArtifactValidationError, MarketDataUnavailable):
                    pass
            engine = _require_engine(app)
            return engine.run_market_forecast(
                horizon_days=request.horizon_days,
                risk=request.risk,
                top_n=request.top_n,
                window_size=request.window_size,
            )
        except HTTPException:
            raise
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get(f"{settings.api_prefix}/market/indices", response_model=MarketIndexResponse)
    def market_indices():
        forecast_engine = _market_forecast_engine(app)
        if forecast_engine is None:
            return _market_index_cache_payload(app)
        try:
            payload = forecast_engine.market_indices_payload()
            if payload.get("indices"):
                return payload
            return _market_index_cache_payload(app)
        except MarketDataUnavailable as exc:
            cache_payload = _market_index_cache_payload(app)
            if cache_payload["indices"]:
                return cache_payload
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(
        f"{settings.api_prefix}/portfolio/simulations",
        response_model=PortfolioSimulationResponse,
    )
    def portfolio_simulation(request: PortfolioSimulationRequest):
        try:
            forecast_engine = _market_forecast_engine(app)
            if forecast_engine is not None:
                try:
                    return forecast_engine.run_portfolio_simulation(
                        amount=request.amount,
                        risk=request.risk,
                        horizon_days=request.horizon_days,
                        selected_tickers=request.selected_tickers,
                        window_size=request.window_size,
                    )
                except (ArtifactValidationError, MarketDataUnavailable):
                    pass
            engine = _require_engine(app)
            return engine.run_portfolio_simulation(
                amount=request.amount,
                risk=request.risk,
                horizon_days=request.horizon_days,
                selected_tickers=request.selected_tickers,
                window_size=request.window_size,
            )
        except HTTPException:
            raise
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get(f"{settings.api_prefix}/data/refresh/status", response_model=RefreshStatusResponse)
    def refresh_status():
        repository = getattr(app.state, "market_repository", None)
        if repository is None:
            return _local_refresh_status_payload(app)
        try:
            return repository.get_latest_refresh_status()
        except MarketDataUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(f"{settings.api_prefix}/inference", response_model=InferenceResponse)
    def inference(request: InferenceRequest):
        try:
            engine = _require_engine(app)
            result = engine.run_inference(
                amount=request.amount,
                risk=request.risk,
                duration=request.duration,
                window_size=request.window_size,
            )
            return {
                "model_version": result.model_version,
                "warnings": result.warnings,
                "summary": result.summary.__dict__,
                "class_allocations": result.class_allocations,
                "asset_allocations": result.asset_allocations,
                "sub_agent_allocations": result.sub_agent_allocations,
                "latest_snapshot": result.latest_snapshot,
                "top_asset_targets": result.top_asset_targets,
                "trade_log": result.trade_log,
            }
        except HTTPException:
            raise
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(f"{settings.api_prefix}/explanations", response_model=ExplanationResponse)
    def explanations(request: ExplanationRequest):
        try:
            engine = _require_engine(app)
            inference_result = engine.run_inference(
                amount=request.amount,
                risk=request.risk,
                duration=request.duration,
                window_size=request.window_size,
            )
            targets = build_explanations(
                engine=engine,
                inference_result=inference_result,
                amount=request.amount,
                risk=request.risk,
                duration=request.duration,
                window_size=request.window_size,
                requested_targets=request.targets,
            )
            return {
                "model_version": inference_result.model_version,
                "warnings": inference_result.warnings,
                "latest_snapshot": inference_result.latest_snapshot,
                "targets": [target.__dict__ for target in targets],
            }
        except HTTPException:
            raise
        except ExplainabilityUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(f"{settings.api_prefix}/backtests", response_model=BacktestResponse)
    def backtests(request: BacktestRequest):
        try:
            engine = _require_engine(app)
            result = engine.run_backtest(
                initial_amount=request.initial_amount,
                risk=request.risk,
                window_size=request.window_size,
                max_steps=request.max_steps or settings.default_backtest_steps,
            )
            return result.__dict__
        except HTTPException:
            raise
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
