from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.app.api.routes.dependencies import (
    _artifact_engine_status,
    _degraded_health_payload,
    _market_health_payload,
    _raise_if_required_supabase_unavailable,
    _require_engine,
)
from backend.app.api.schemas import HealthResponse, ModelsResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    app = request.app
    engine = getattr(app.state, "engine", None)
    if engine is None:
        payload = _degraded_health_payload(app)
        _raise_if_required_supabase_unavailable(payload)
        return payload
    payload = engine.health_payload()
    payload["ready"] = True
    payload["market_data"] = _market_health_payload(app)
    payload["artifact_engine"] = _artifact_engine_status(app)
    _raise_if_required_supabase_unavailable(payload)
    return payload


@router.get("/models", response_model=ModelsResponse)
def models(request: Request):
    try:
        engine = _require_engine(request.app)
        return engine.model_payload()
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail=str(exc)) from exc
