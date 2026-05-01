from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
import json
import os
import sys
import uuid
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import get_settings
from backend.app.market.forecasting import SupabaseForecastEngine
from backend.app.market.repository import (
    InMemoryMarketDataRepository,
    SupabaseMarketDataRepository,
)
from backend.app.ml.errors import ArtifactValidationError
from offline.market_data_providers import MarketDataProvider, build_provider


DEFAULT_HORIZONS = (30, 90, 180, 300)


def parse_horizons(value: str) -> list[int]:
    horizons = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not horizons:
        raise ValueError("At least one forecast horizon is required")
    return horizons


def load_universe(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    assets = payload.get("assets", [])
    if not isinstance(assets, list) or not assets:
        raise ValueError(f"No assets found in universe config: {path}")
    normalized = []
    for row in assets:
        normalized.append(
            {
                "ticker": row["ticker"].strip().upper(),
                "asset_class": row["asset_class"],
                "display_name": row.get("display_name") or row["ticker"],
                "exchange": row.get("exchange"),
                "currency": row.get("currency") or "USD",
                "sector": row.get("sector"),
                "industry": row.get("industry"),
                "country": row.get("country"),
                "provider_symbol": row.get("provider_symbol") or row["ticker"],
                "benchmark_group": row.get("benchmark_group"),
                "min_history_days": int(row.get("min_history_days") or 252),
                "active": bool(row.get("active", True)),
                "start_date": row.get("start_date") or payload.get("default_start_date") or "2018-01-01",
            }
        )
    return normalized


def load_market_indices(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    indices = payload.get("indices", [])
    if not isinstance(indices, list):
        raise ValueError(f"Invalid market index config: {path}")
    normalized = []
    for row in indices:
        normalized.append(
            {
                "symbol": row["symbol"].strip().upper(),
                "label": row.get("label") or row["symbol"],
                "display_name": row.get("display_name") or row.get("label") or row["symbol"],
                "provider_symbol": row.get("provider_symbol") or row["symbol"],
                "currency": row.get("currency") or "USD",
                "display_order": int(row.get("display_order") or 0),
            }
        )
    return normalized


def universe_upsert_rows(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in asset.items()
            if key
            in {
                "ticker",
                "asset_class",
                "display_name",
                "exchange",
                "currency",
                "sector",
                "industry",
                "country",
                "provider_symbol",
                "benchmark_group",
                "min_history_days",
                "active",
            }
        }
        for asset in assets
    ]


def build_repository_from_env():
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if url.startswith("memory://"):
        return InMemoryMarketDataRepository()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    return SupabaseMarketDataRepository(url=url, service_role_key=key)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _group_by_ticker(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["ticker"]].append(row)
    return dict(grouped)


def _latest_row_date(rows: list[dict[str, Any]]) -> date | None:
    dates = [date.fromisoformat(row["date"]) for row in rows if row.get("date")]
    return max(dates) if dates else None


def _freshness_cutoff(end_date: str, freshness_days: int) -> date:
    return date.fromisoformat(end_date) - timedelta(days=max(int(freshness_days), 1))


def _record_item(
    items: list[dict[str, Any]],
    *,
    run_id: str,
    ticker: str,
    stage: str,
    status: str,
    rows_written: int = 0,
    error: str | None = None,
) -> None:
    items.append(
        {
            "run_id": run_id,
            "ticker": ticker,
            "stage": stage,
            "status": status,
            "rows_written": int(rows_written),
            "error": error,
        }
    )


def run_refresh(
    *,
    repository,
    provider: MarketDataProvider,
    assets: list[dict[str, Any]],
    market_indices: list[dict[str, Any]] | None = None,
    start_date: str,
    end_date: str,
    horizons: list[int],
    window_size: int,
    freshness_days: int = 10,
    dry_run: bool = False,
) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    started_at = _now_iso()
    item_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "run_id": run_id,
        "provider": provider.name,
        "dry_run": dry_run,
        "start_date": start_date,
        "end_date": end_date,
        "universe_count": len(assets),
        "market_index_count": len(market_indices or []),
        "rows": {},
        "failures": [],
    }
    if not dry_run and hasattr(repository, "validate_schema"):
        repository.validate_schema()

    run_row = {
        "id": run_id,
        "provider": provider.name,
        "status": "running",
        "started_at": started_at,
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "rows_inserted": 0,
        "rows_updated": 0,
        "metadata": {
            "horizons": horizons,
            "window_size": window_size,
            "market_index_count": len(market_indices or []),
        },
    }
    if not dry_run:
        repository.upsert_refresh_run(run_row)

    total_written = 0
    try:
        universe_rows = universe_upsert_rows(assets)
        if not dry_run:
            total_written += repository.upsert_universe(universe_rows)
            if hasattr(repository, "deactivate_missing_universe"):
                total_written += repository.deactivate_missing_universe(
                    {row["ticker"] for row in universe_rows}
                )
        report["rows"]["asset_universe"] = len(universe_rows)

        active_assets = [asset for asset in assets if asset.get("active", True)]
        try:
            ohlcv_rows = provider.fetch_daily_ohlcv(
                active_assets,
                start_date=start_date,
                end_date=end_date,
                freshness_days=freshness_days,
            )
            grouped_ohlcv = _group_by_ticker(ohlcv_rows)
            if not dry_run:
                total_written += repository.upsert_ohlcv(ohlcv_rows)
            report["rows"]["market_ohlcv_daily"] = len(ohlcv_rows)
            cutoff = _freshness_cutoff(end_date, freshness_days)
            for asset in active_assets:
                ticker_rows = grouped_ohlcv.get(asset["ticker"], [])
                latest_date = _latest_row_date(ticker_rows)
                is_fresh = latest_date is not None and latest_date >= cutoff
                _record_item(
                    item_rows,
                    run_id=run_id,
                    ticker=asset["ticker"],
                    stage="ohlcv",
                    status="completed" if is_fresh else "failed",
                    rows_written=len(ticker_rows),
                    error=(
                        None
                        if is_fresh
                        else (
                            "Provider returned no OHLCV rows"
                            if not ticker_rows
                            else f"Latest OHLCV row {latest_date.isoformat()} is older than freshness cutoff {cutoff.isoformat()}"
                        )
                    ),
                )
        except Exception as exc:
            report["failures"].append({"stage": "ohlcv", "error": str(exc)})
            for asset in active_assets:
                _record_item(
                    item_rows,
                    run_id=run_id,
                    ticker=asset["ticker"],
                    stage="ohlcv",
                    status="failed",
                    error=str(exc),
                )

        try:
            profile_rows = provider.fetch_company_profiles(active_assets)
            grouped_profiles = _group_by_ticker(profile_rows)
            if not dry_run:
                total_written += repository.upsert_profiles(profile_rows)
            report["rows"]["asset_profile_snapshots"] = len(profile_rows)
            for asset in active_assets:
                ticker_rows = grouped_profiles.get(asset["ticker"], [])
                _record_item(
                    item_rows,
                    run_id=run_id,
                    ticker=asset["ticker"],
                    stage="profile",
                    status="completed" if ticker_rows else "failed",
                    rows_written=len(ticker_rows),
                    error=None if ticker_rows else "Provider returned no profile row",
                )
        except Exception as exc:
            report["failures"].append({"stage": "profile", "error": str(exc)})
            for asset in active_assets:
                _record_item(
                    item_rows,
                    run_id=run_id,
                    ticker=asset["ticker"],
                    stage="profile",
                    status="failed",
                    error=str(exc),
                )

        try:
            macro_rows = provider.fetch_macro_observations(
                start_date=start_date,
                end_date=end_date,
            )
            if not dry_run:
                total_written += repository.upsert_macro(macro_rows)
            report["rows"]["macro_observations"] = len(macro_rows)
            _record_item(
                item_rows,
                run_id=run_id,
                ticker="__macro__",
                stage="macro",
                status="completed",
                rows_written=len(macro_rows),
            )
        except Exception as exc:
            report["failures"].append({"stage": "macro", "error": str(exc)})
            _record_item(
                item_rows,
                run_id=run_id,
                ticker="__macro__",
                stage="macro",
                status="failed",
                error=str(exc),
            )

        index_rows = []
        if market_indices:
            try:
                index_rows = provider.fetch_market_indices(
                    market_indices,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not dry_run and hasattr(repository, "upsert_market_indices"):
                    total_written += repository.upsert_market_indices(index_rows)
                report["rows"]["market_index_snapshots"] = len(index_rows)
                by_symbol = {row["symbol"]: row for row in index_rows}
                for index in market_indices:
                    symbol = index["symbol"]
                    row = by_symbol.get(symbol)
                    _record_item(
                        item_rows,
                        run_id=run_id,
                        ticker=symbol,
                        stage="market_index",
                        status="completed" if row else "failed",
                        rows_written=1 if row else 0,
                        error=None if row else "Provider returned no index snapshot",
                    )
            except Exception as exc:
                report["failures"].append({"stage": "market_index", "error": str(exc)})
                for index in market_indices:
                    _record_item(
                        item_rows,
                        run_id=run_id,
                        ticker=index["symbol"],
                        stage="market_index",
                        status="failed",
                        error=str(exc),
                    )
        else:
            report["rows"]["market_index_snapshots"] = 0

        forecast_rows = []
        if not dry_run:
            forecast_engine = SupabaseForecastEngine(repository, get_settings())
            for asset in active_assets:
                for horizon in horizons:
                    try:
                        forecast = forecast_engine.build_ticker_forecast(
                            ticker=asset["ticker"],
                            horizon_days=horizon,
                            window_size=window_size,
                            prefer_snapshot=False,
                        )
                        forecast_rows.append(forecast_engine.forecast_snapshot_row(forecast))
                    except ArtifactValidationError as exc:
                        _record_item(
                            item_rows,
                            run_id=run_id,
                            ticker=asset["ticker"],
                            stage=f"forecast_{horizon}",
                            status="failed",
                            error=str(exc),
                        )
            total_written += repository.upsert_forecasts(forecast_rows)
        report["rows"]["forecast_snapshots"] = len(forecast_rows)

        if not dry_run:
            repository.upsert_refresh_run_items(item_rows)
        failed_items = [item for item in item_rows if item["status"] == "failed"]
        status = "completed"
        if report["failures"] or failed_items:
            status = "partial" if total_written > 0 else "failed"
        final_run = {
            **run_row,
            "status": status,
            "finished_at": _now_iso(),
            "rows_inserted": int(total_written),
            "error": json.dumps(report["failures"]) if report["failures"] else None,
        }
        if not dry_run:
            repository.upsert_refresh_run(final_run)
        report["status"] = status
        report["items"] = item_rows
        report["rows_written"] = total_written
        return report
    except Exception as exc:
        final_run = {
            **run_row,
            "status": "failed",
            "finished_at": _now_iso(),
            "rows_inserted": int(total_written),
            "error": str(exc),
        }
        if not dry_run:
            repository.upsert_refresh_run(final_run)
            if item_rows:
                repository.upsert_refresh_run_items(item_rows)
        raise


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Refresh Foresight Supabase market data.")
    parser.add_argument(
        "--universe-path",
        default=str(REPO_ROOT / "config" / "asset_universe.v1.json"),
    )
    parser.add_argument(
        "--macro-path",
        default=str(REPO_ROOT / "datasets" / "raw" / "macroeconomic_data_2010_2024.csv"),
    )
    parser.add_argument(
        "--indices-path",
        default=str(REPO_ROOT / "config" / "market_indices.v1.json"),
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("STOCKIFY_MARKET_DATA_PROVIDER", "yfinance"),
    )
    parser.add_argument("--mode", choices=["incremental", "full"], default="incremental")
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=(date.today() + timedelta(days=1)).isoformat())
    parser.add_argument("--horizons", default=",".join(str(value) for value in DEFAULT_HORIZONS))
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--freshness-days", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    assets = load_universe(Path(args.universe_path))
    indices = load_market_indices(Path(args.indices_path))
    if args.start_date:
        start_date = args.start_date
    elif args.mode == "full":
        start_date = min(asset["start_date"] for asset in assets)
    else:
        start_date = (date.fromisoformat(args.end_date) - timedelta(days=args.lookback_days)).isoformat()

    if args.dry_run and (not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_ROLE_KEY")):
        repository = InMemoryMarketDataRepository()
    else:
        repository = build_repository_from_env()
    provider = build_provider(args.provider, macro_path=Path(args.macro_path))
    report = run_refresh(
        repository=repository,
        provider=provider,
        assets=assets,
        market_indices=indices,
        start_date=start_date,
        end_date=args.end_date,
        horizons=parse_horizons(args.horizons),
        window_size=args.window_size,
        freshness_days=args.freshness_days,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
