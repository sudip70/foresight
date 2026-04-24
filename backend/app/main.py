from __future__ import annotations

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
    ModelsResponse,
    PortfolioSimulationRequest,
    PortfolioSimulationResponse,
    TickerForecastRequest,
    TickerForecastResponse,
    UniverseResponse,
)
from backend.app.core.config import get_settings
from backend.app.ml.artifacts import ArtifactValidationError
from backend.app.ml.explainability import ExplainabilityUnavailable, build_explanations
from backend.app.ml.pipeline import get_engine


def _load_engine(strict_validation: bool):
    settings = get_settings()
    return get_engine(
        str(settings.artifact_root),
        str(settings.dataset_root),
        strict_validation,
        settings.surrogate_sample_size,
        settings.surrogate_fidelity_threshold,
        settings.top_asset_target_count,
        settings.default_backtest_steps,
        settings.meta_max_asset_weight,
        settings.meta_max_stock_weight,
        settings.meta_max_crypto_weight,
        settings.meta_max_etf_weight,
        settings.meta_max_cash_weight,
        settings.meta_min_expected_daily_return,
        settings.meta_cash_enabled,
        settings.meta_cash_annual_return,
    )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.project_name,
        version="1.0.0",
        description="RL allocation, SHAP explainability, and backtesting API for Stockify.",
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
        try:
            engine = _load_engine(False)
            return engine.health_payload()
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get(f"{settings.api_prefix}/models", response_model=ModelsResponse)
    def models():
        try:
            engine = _load_engine(False)
            return engine.model_payload()
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get(f"{settings.api_prefix}/universe", response_model=UniverseResponse)
    def universe():
        try:
            engine = _load_engine(False)
            return engine.universe_payload()
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(f"{settings.api_prefix}/forecasts/ticker", response_model=TickerForecastResponse)
    def ticker_forecast(request: TickerForecastRequest):
        try:
            engine = _load_engine(request.strict_validation)
            return engine.run_ticker_forecast(
                ticker=request.ticker,
                horizon_days=request.horizon_days,
                window_size=request.window_size,
            )
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(f"{settings.api_prefix}/forecasts/market", response_model=MarketForecastResponse)
    def market_forecast(request: MarketForecastRequest):
        try:
            engine = _load_engine(request.strict_validation)
            return engine.run_market_forecast(
                horizon_days=request.horizon_days,
                risk=request.risk,
                top_n=request.top_n,
                window_size=request.window_size,
            )
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(
        f"{settings.api_prefix}/portfolio/simulations",
        response_model=PortfolioSimulationResponse,
    )
    def portfolio_simulation(request: PortfolioSimulationRequest):
        try:
            engine = _load_engine(request.strict_validation)
            return engine.run_portfolio_simulation(
                amount=request.amount,
                risk=request.risk,
                horizon_days=request.horizon_days,
                selected_tickers=request.selected_tickers,
                window_size=request.window_size,
            )
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(f"{settings.api_prefix}/inference", response_model=InferenceResponse)
    def inference(request: InferenceRequest):
        try:
            engine = _load_engine(request.strict_validation)
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
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(f"{settings.api_prefix}/explanations", response_model=ExplanationResponse)
    def explanations(request: ExplanationRequest):
        try:
            engine = _load_engine(request.strict_validation)
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
        except ExplainabilityUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(f"{settings.api_prefix}/backtests", response_model=BacktestResponse)
    def backtests(request: BacktestRequest):
        try:
            engine = _load_engine(request.strict_validation)
            result = engine.run_backtest(
                initial_amount=request.initial_amount,
                risk=request.risk,
                window_size=request.window_size,
                max_steps=request.max_steps or settings.default_backtest_steps,
            )
            return result.__dict__
        except ArtifactValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
