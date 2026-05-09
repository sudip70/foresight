from __future__ import annotations

from contextlib import asynccontextmanager
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import api_routers
from backend.app.api.routes.dependencies import (
    _load_engine,
    _load_market_repository,
    _should_preload_artifact_engine,
    _start_artifact_engine_background_load,
)
from backend.app.core.config import get_settings
from backend.app.market.forecasting import SupabaseForecastEngine
from backend.app.market.index_refresh import (
    fetch_market_index_history,
    fetch_market_index_history_from_repository,
    fetch_market_index_snapshots,
    fetch_market_index_snapshots_from_repository,
    refresh_market_index_snapshots,
)
from backend.app.market.repository import build_market_repository


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = None
        app.state.engine_error = None
        app.state.engine_loading = False
        app.state.engine_lock = threading.Lock()
        app.state.engine_loader_thread = None
        app.state.market_repository = None
        app.state.market_repository_error = None
        app.state.forecast_engine = None
        app.state.market_index_rows = []
        app.state.market_index_refresh_result = None
        app.state.market_index_refresh_error = None
        app.state.market_index_history_cache = {}
        try:
            repository = _load_market_repository()
            if repository is not None and hasattr(repository, "validate_schema"):
                repository.validate_schema()
            app.state.market_repository = repository
            if repository is not None:
                app.state.forecast_engine = SupabaseForecastEngine(
                    repository,
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
        if settings.load_artifact_engine:
            try:
                app.state.engine = _load_engine()
            except Exception as exc:  # pragma: no cover - defensive startup path
                app.state.engine_error = exc
        elif settings.lazy_load_artifact_engine:
            app.state.engine_error = None
        else:
            app.state.engine_error = "Artifact engine disabled by FORESIGHT_LOAD_ARTIFACT_ENGINE=false"
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

    @app.middleware("http")
    async def trigger_artifact_engine_load(request, call_next):
        if _should_preload_artifact_engine(request.url.path):
            _start_artifact_engine_background_load(app)
        return await call_next(request)

    for router in api_routers:
        app.include_router(router, prefix=settings.api_prefix)

    return app


app = create_app()
