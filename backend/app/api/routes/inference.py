from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.app.api.routes.dependencies import _require_engine
from backend.app.api.schemas import (
    ExplanationRequest,
    ExplanationResponse,
    InferenceRequest,
    InferenceResponse,
)
from backend.app.ml.errors import ArtifactValidationError, ExplainabilityUnavailable

router = APIRouter(tags=["inference"])


@router.post("/inference", response_model=InferenceResponse)
def inference(request_body: InferenceRequest, request: Request):
    try:
        engine = _require_engine(request.app)
        result = engine.run_inference(
            amount=request_body.amount,
            risk=request_body.risk,
            duration=request_body.duration,
            window_size=request_body.window_size,
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


@router.post("/explanations", response_model=ExplanationResponse)
def explanations(request_body: ExplanationRequest, request: Request):
    try:
        from backend.app.ml.explainability import build_explanations

        engine = _require_engine(request.app)
        inference_result = engine.run_inference(
            amount=request_body.amount,
            risk=request_body.risk,
            duration=request_body.duration,
            window_size=request_body.window_size,
        )
        targets = build_explanations(
            engine=engine,
            inference_result=inference_result,
            amount=request_body.amount,
            risk=request_body.risk,
            duration=request_body.duration,
            window_size=request_body.window_size,
            requested_targets=request_body.targets,
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
