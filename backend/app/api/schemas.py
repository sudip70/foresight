from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    artifact_root: str
    dataset_root: str
    dependencies: dict[str, bool]
    artifacts: dict[str, dict[str, Any]]
    meta_model: dict[str, Any]
    ready: bool | None = None
    error: str | None = None
    market_data: dict[str, Any] | None = None
    artifact_engine: dict[str, Any] | None = None


class ModelsResponse(BaseModel):
    supported_asset_classes: list[str]
    asset_agents: list[dict[str, Any]]
    meta_agent: dict[str, Any]
    feature_groups: dict[str, dict[str, int]]
    explainability: dict[str, Any]


class UniverseResponse(BaseModel):
    latest_date: str
    supported_asset_classes: list[str]
    asset_classes: list[dict[str, Any]]
    tickers: list[dict[str, Any]]
    disclaimer: str
    source: str | None = None


class TickerProfileResponse(BaseModel):
    ticker: str
    asset_class: str
    display_name: str | None = None
    as_of_date: str | None = None
    data_as_of: str | None = None
    source: str
    fields: dict[str, Any]


class TickerForecastRequest(BaseModel):
    ticker: str = Field(min_length=1)
    horizon_days: int = Field(default=300, ge=1)
    window_size: int = Field(default=60, ge=2)
    strict_validation: bool = True


class TickerForecastResponse(BaseModel):
    ticker: str
    asset_class: str
    latest_date: str
    forecast_start_date: str | None = None
    latest_price: float
    horizon_days: int
    historical_prices: list[dict[str, Any]]
    forecast_paths: dict[str, list[dict[str, Any]]]
    target_prices: dict[str, float]
    returns: dict[str, float]
    risk_metrics: dict[str, float]
    confidence: float
    confidence_label: str
    risk_label: str
    opportunity_score: float
    return_estimator: dict[str, Any]
    literacy: dict[str, str]
    plain_language: str
    data_as_of: str | None = None
    source: str | None = None
    snapshot_used: bool | None = None


class MarketForecastRequest(BaseModel):
    horizon_days: int = Field(default=300, ge=1)
    risk: float = Field(default=0.5, ge=0.0, le=1.0)
    top_n: int = Field(default=10, ge=1, le=50)
    window_size: int = Field(default=60, ge=2)
    strict_validation: bool = True


class MarketForecastResponse(BaseModel):
    horizon_days: int
    risk: float
    ranked_tickers: list[dict[str, Any]]
    highlights: dict[str, Any]
    macro_snapshot: dict[str, Any]
    disclaimer: str
    source: str | None = None


class MarketIndexResponse(BaseModel):
    source: str
    as_of_date: str | None = None
    indices: list[dict[str, Any]]
    disclaimer: str


class MarketIndexHistoryResponse(BaseModel):
    source: str
    symbol: str
    label: str
    display_name: str | None = None
    provider_symbol: str
    currency: str
    range: str
    lookback_days: int
    as_of_date: str | None = None
    history: list[dict[str, Any]]
    summary: dict[str, Any]
    disclaimer: str


class PortfolioSimulationRequest(BaseModel):
    amount: float = Field(default=10000.0, gt=0)
    risk: float = Field(default=0.5, ge=0.0, le=1.0)
    horizon_days: int = Field(default=300, ge=1)
    selected_tickers: list[str] | None = None
    window_size: int = Field(default=60, ge=2)
    strict_validation: bool = True


class PortfolioSimulationResponse(BaseModel):
    amount: float
    risk: float
    horizon_days: int
    method: str
    summary: dict[str, float]
    asset_allocations: list[dict[str, Any]]
    class_allocations: list[dict[str, Any]]
    trade_plan: list[dict[str, Any]]
    source_forecasts: list[dict[str, Any]]
    warnings: list[str]


class InferenceRequest(BaseModel):
    amount: float = Field(default=10000.0, gt=0)
    risk: float = Field(default=0.5, ge=0.0, le=1.0)
    duration: int = Field(default=30, ge=1)
    window_size: int = Field(default=60, ge=2)
    strict_validation: bool = True


class InferenceResponse(BaseModel):
    model_version: str
    warnings: list[str]
    summary: dict[str, float]
    class_allocations: list[dict[str, Any]]
    asset_allocations: list[dict[str, Any]]
    sub_agent_allocations: dict[str, list[dict[str, Any]]]
    latest_snapshot: dict[str, Any]
    top_asset_targets: list[str]
    trade_log: list[dict[str, Any]]


class ExplanationRequest(InferenceRequest):
    targets: list[str] | None = None


class ExplanationTargetResponse(BaseModel):
    target: str
    available: bool
    fidelity: float
    grouped_contributions: dict[str, float]
    top_positive_drivers: list[dict[str, Any]]
    top_negative_drivers: list[dict[str, Any]]
    plain_language: str | None


class ExplanationResponse(BaseModel):
    model_version: str
    warnings: list[str]
    latest_snapshot: dict[str, Any]
    targets: list[ExplanationTargetResponse]


class BacktestRequest(BaseModel):
    initial_amount: float = Field(default=10000.0, gt=0)
    risk: float = Field(default=0.5, ge=0.0, le=1.0)
    window_size: int = Field(default=60, ge=2)
    max_steps: int | None = Field(default=None, ge=1)
    strict_validation: bool = True


class BacktestResponse(BaseModel):
    summary_metrics: dict[str, float]
    equity_curve: list[dict[str, Any]]
    drawdown_curve: list[dict[str, Any]]
    trade_log: list[dict[str, Any]]
    warnings: list[str]


class RefreshStatusResponse(BaseModel):
    configured: bool
    source: str
    latest_run: dict[str, Any] | None = None
    failed_items: list[dict[str, Any]]
    stale_tickers: list[str]
    latest_market_date: str | None = None
    message: str | None = None
    asset_count: int | None = None
    market_index_refresh: dict[str, Any] | None = None
    checked_at: str | None = None
    today: str | None = None
