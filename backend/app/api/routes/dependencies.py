from __future__ import annotations

import json
import threading

from fastapi import FastAPI, HTTPException

from backend.app.core.config import REPO_ROOT, get_settings
from backend.app.market.index_refresh import fetch_market_index_snapshots_from_repository
from backend.app.market.repository import MarketDataUnavailable, empty_refresh_status


def _load_engine():
    from backend.app.ml.pipeline import get_engine

    return get_engine(get_settings(), strict_validation=True)


def _load_market_repository():
    from backend.app import main as app_main

    return app_main.build_market_repository(get_settings())


def _market_health_payload(app: FastAPI) -> dict | None:
    repository = getattr(app.state, "market_repository", None)
    if repository is None:
        return None
    return repository.health_payload()


def _artifact_engine_status(app: FastAPI) -> dict:
    settings = get_settings()
    engine = getattr(app.state, "engine", None)
    loading = bool(getattr(app.state, "engine_loading", False))
    error = getattr(app.state, "engine_error", None)
    if engine is not None:
        status = "ready"
    elif loading:
        status = "loading"
    elif settings.lazy_load_artifact_engine:
        status = "error" if error else "not_started"
    elif settings.load_artifact_engine:
        status = "error" if error else "not_loaded"
    else:
        status = "disabled"
    return {
        "status": status,
        "startup_enabled": settings.load_artifact_engine,
        "lazy_enabled": settings.lazy_load_artifact_engine,
        "error": str(error) if error and status in {"error", "disabled"} else None,
    }


def _start_artifact_engine_background_load(app: FastAPI) -> bool:
    settings = get_settings()
    if not settings.lazy_load_artifact_engine:
        return False
    if getattr(app.state, "engine", None) is not None:
        return False

    lock = getattr(app.state, "engine_lock", None)
    if lock is None:
        return False

    with lock:
        if getattr(app.state, "engine", None) is not None:
            return False
        if getattr(app.state, "engine_loading", False):
            return False

        app.state.engine_loading = True
        app.state.engine_error = "Artifact engine is loading in background"

        def load() -> None:
            try:
                engine = _load_engine()
            except Exception as exc:  # pragma: no cover - background defensive path
                with lock:
                    app.state.engine = None
                    app.state.engine_error = exc
                    app.state.engine_loading = False
            else:
                with lock:
                    app.state.engine = engine
                    app.state.engine_error = None
                    app.state.engine_loading = False

        thread = threading.Thread(
            target=load,
            name="foresight-artifact-engine-loader",
            daemon=True,
        )
        app.state.engine_loader_thread = thread
        thread.start()
        return True


def _should_preload_artifact_engine(path: str) -> bool:
    api_prefix = get_settings().api_prefix.rstrip("/")
    normalized = path.rstrip("/") or "/"
    artifact_paths = {
        f"{api_prefix}/models",
        f"{api_prefix}/inference",
        f"{api_prefix}/explanations",
        f"{api_prefix}/backtests",
    }
    return normalized in artifact_paths


def _degraded_health_payload(app: FastAPI) -> dict:
    settings = get_settings()
    error = getattr(app.state, "engine_error", None)
    market_data = _market_health_payload(app)
    repository_error = getattr(app.state, "market_repository_error", None)
    if market_data is None and repository_error is not None:
        market_data = {
            "configured": bool(settings.supabase_url and settings.supabase_service_role_key),
            "status": "unavailable",
            "source": "supabase",
            "error": str(repository_error),
        }
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
        "artifact_engine": _artifact_engine_status(app),
    }


def _raise_if_required_supabase_unavailable(payload: dict) -> None:
    settings = get_settings()
    market_data = payload.get("market_data") or {}
    if settings.require_supabase and market_data.get("status") != "ok":
        raise HTTPException(
            status_code=503,
            detail=market_data.get("error") or "Supabase market data is required but unavailable",
        )


def _require_engine(app: FastAPI):
    engine = getattr(app.state, "engine", None)
    if engine is None:
        _start_artifact_engine_background_load(app)
        error = getattr(app.state, "engine_error", None)
        loading = bool(getattr(app.state, "engine_loading", False))
        detail = "Foresight engine is not ready"
        headers = None
        if loading:
            detail = "Foresight engine is loading in the background; retry in a few seconds"
            headers = {"Retry-After": "15"}
        if error is not None:
            detail = f"{detail}: {error}"
        raise HTTPException(status_code=503, detail=detail, headers=headers)
    return engine


def _market_forecast_engine(app: FastAPI):
    return getattr(app.state, "forecast_engine", None)


def _allow_artifact_fallback() -> bool:
    settings = get_settings()
    return settings.load_artifact_engine and not settings.require_supabase


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


def _market_index_live_payload(app: FastAPI) -> dict:
    from backend.app import main as app_main

    refresh_result = app_main.fetch_market_index_snapshots(
        get_settings(),
        repository=getattr(app.state, "market_repository", None),
    )
    app.state.market_index_refresh_result = refresh_result
    app.state.market_index_rows = refresh_result.get("rows", []) or []
    app.state.market_index_refresh_error = None
    return _market_index_cache_payload(app)


def _market_index_repository_proxy_payload(app: FastAPI) -> dict:
    repository = getattr(app.state, "market_repository", None)
    if repository is None:
        raise MarketDataUnavailable("Supabase market data is not available")
    refresh_result = fetch_market_index_snapshots_from_repository(
        get_settings(),
        repository=repository,
    )
    app.state.market_index_refresh_result = refresh_result
    app.state.market_index_rows = refresh_result.get("rows", []) or []
    app.state.market_index_refresh_error = None
    payload = _market_index_cache_payload(app)
    payload["source"] = "supabase_proxy"
    payload["disclaimer"] = (
        "Index cards use Supabase ETF proxy history when direct index snapshots "
        "or live provider data are unavailable."
    )
    return payload


def _market_index_proxy_or_503(app: FastAPI, primary_error: Exception) -> dict:
    try:
        return _market_index_repository_proxy_payload(app)
    except Exception as proxy_error:
        detail = str(primary_error) or "Market index data is unavailable"
        proxy_detail = str(proxy_error)
        if proxy_detail and proxy_detail != detail:
            detail = f"{detail}; Supabase proxy fallback unavailable: {proxy_detail}"
        raise HTTPException(status_code=503, detail=detail) from proxy_error


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
    data = engine.ticker_raw_data(ticker)
    metadata = _asset_universe_metadata().get(data["canonical"].upper(), {})
    ohlcv = data["ohlcv"]
    ohlcv_window = data["ohlcv_window"]
    return {
        "ticker": data["canonical"],
        "asset_class": data["asset_class"],
        "display_name": metadata.get("display_name") or data["canonical"],
        "as_of_date": None,
        "data_as_of": data["latest_date"],
        "source": "local_artifacts",
        "fields": {
            "bid": None,
            "ask": None,
            "last_sale": data["price"],
            "open": float(ohlcv[0]),
            "high": float(ohlcv[1]),
            "low": float(ohlcv[2]),
            "exchange": metadata.get("exchange"),
            "sector": metadata.get("sector"),
            "industry": metadata.get("industry"),
            "country": metadata.get("country"),
            "benchmark_group": metadata.get("benchmark_group"),
            "provider_symbol": metadata.get("provider_symbol"),
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
