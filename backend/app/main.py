from __future__ import annotations

from contextlib import asynccontextmanager
import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    fetch_market_index_snapshots,
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
            def _do_index_refresh(
                _app=app,
                _settings=settings,
                _repository=app.state.market_repository,
            ) -> None:
                try:
                    result = refresh_market_index_snapshots(_settings, repository=_repository)
                    _app.state.market_index_refresh_result = result
                    _app.state.market_index_rows = result.get("rows", [])
                except Exception as exc:  # pragma: no cover - provider/network defensive path
                    _app.state.market_index_refresh_error = exc

            threading.Thread(
                target=_do_index_refresh,
                name="foresight-index-refresh",
                daemon=True,
            ).start()
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
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _rate_limit_store: dict[str, tuple[int, float]] = {}
    _RATE_LIMIT_MAX = 60
    _RATE_LIMIT_WINDOW = 60.0
    _RATE_LIMIT_CLEANUP_INTERVAL = 300.0
    _rate_limit_last_cleanup = [time.monotonic()]
    _health_path = f"{settings.api_prefix}/health"

    @app.middleware("http")
    async def rate_limit_middleware(request, call_next):
        if request.url.path.rstrip("/") == _health_path:
            return await call_next(request)
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        if now - _rate_limit_last_cleanup[0] >= _RATE_LIMIT_CLEANUP_INTERVAL:
            stale = [k for k, (_, ws) in _rate_limit_store.items() if now - ws >= _RATE_LIMIT_WINDOW]
            for k in stale:
                del _rate_limit_store[k]
            _rate_limit_last_cleanup[0] = now
        count, window_start = _rate_limit_store.get(ip, (0, now))
        if now - window_start >= _RATE_LIMIT_WINDOW:
            count, window_start = 0, now
        count += 1
        _rate_limit_store[ip] = (count, window_start)
        if count > _RATE_LIMIT_MAX:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait before retrying."},
                headers={"Retry-After": "60"},
            )
        return await call_next(request)

    @app.middleware("http")
    async def trigger_artifact_engine_load(request, call_next):
        if _should_preload_artifact_engine(request.url.path):
            _start_artifact_engine_background_load(app)
        return await call_next(request)

    for router in api_routers:
        app.include_router(router, prefix=settings.api_prefix)

    return app


app = create_app()
