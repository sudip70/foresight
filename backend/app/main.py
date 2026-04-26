from __future__ import annotations

from contextlib import asynccontextmanager

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
from backend.app.core.config import get_settings
from backend.app.market.forecasting import SupabaseForecastEngine
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


def _artifact_profile_payload(engine, ticker: str) -> dict:
    asset_class, index, canonical = engine._resolve_ticker(ticker)
    context = engine._combined_context["assets"][asset_class]
    latest_index = engine._combined_context["aligned_rows"] - 1
    latest_date = str(engine._combined_context["dates"][latest_index])
    price = float(context["risky_prices"][latest_index, index])
    ohlcv = context["ohlcv"][latest_index, index]
    return {
        "ticker": canonical,
        "asset_class": asset_class,
        "display_name": canonical,
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
            "exchange": None,
            "market_cap": None,
            "pe_ratio": None,
            "fifty_two_week_high": None,
            "fifty_two_week_low": None,
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
        try:
            app.state.market_repository = _load_market_repository()
            if app.state.market_repository is not None:
                app.state.forecast_engine = SupabaseForecastEngine(
                    app.state.market_repository,
                    settings,
                )
        except Exception as exc:  # pragma: no cover - defensive startup path
            app.state.market_repository_error = exc
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
            return {
                "source": "local_artifacts",
                "as_of_date": None,
                "indices": [],
                "disclaimer": "Market index data is available when Supabase market data is configured.",
            }
        try:
            return forecast_engine.market_indices_payload()
        except MarketDataUnavailable as exc:
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
            return empty_refresh_status()
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
