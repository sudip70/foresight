from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    artifact_root: str
    dataset_root: str
    dependencies: dict[str, bool]
    artifacts: dict[str, dict[str, Any]]
    meta_model: dict[str, Any]


class ModelsResponse(BaseModel):
    supported_asset_classes: list[str]
    asset_agents: list[dict[str, Any]]
    meta_agent: dict[str, Any]
    feature_groups: dict[str, dict[str, int]]
    explainability: dict[str, Any]


class InferenceRequest(BaseModel):
    amount: float = Field(default=10000.0, gt=0)
    risk: float = Field(default=0.5, ge=0.0, le=1.0)
    duration: int = Field(default=30, ge=1)
    window_size: int = Field(default=60, ge=2)
    strict_validation: bool = False


class InferenceResponse(BaseModel):
    model_version: str
    warnings: list[str]
    summary: dict[str, float]
    class_allocations: list[dict[str, Any]]
    asset_allocations: list[dict[str, Any]]
    sub_agent_allocations: dict[str, list[dict[str, Any]]]
    latest_snapshot: dict[str, Any]
    top_asset_targets: list[str]


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
    strict_validation: bool = False


class BacktestResponse(BaseModel):
    summary_metrics: dict[str, float]
    equity_curve: list[dict[str, Any]]
    drawdown_curve: list[dict[str, Any]]
    warnings: list[str]

