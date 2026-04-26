from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import uuid
from typing import Any, Protocol

import httpx

from backend.app.core.config import Settings


FORECAST_METHOD_VERSION = "stat-scenarios-v1"
REQUIRED_SUPABASE_TABLES = (
    "asset_universe",
    "market_ohlcv_daily",
    "asset_profile_snapshots",
    "macro_observations",
    "forecast_snapshots",
    "refresh_runs",
    "refresh_run_items",
)


class MarketDataUnavailable(RuntimeError):
    """Raised when the configured market data store cannot serve a request."""


class MarketDataRepository(Protocol):
    def health_payload(self) -> dict[str, Any]: ...

    def list_universe(self) -> list[dict[str, Any]]: ...

    def ticker_exists(self, ticker: str) -> bool: ...

    def ticker_metadata(self, ticker: str) -> dict[str, Any] | None: ...

    def coverage_for_ticker(self, ticker: str) -> dict[str, Any]: ...

    def get_ohlcv_history(self, ticker: str) -> list[dict[str, Any]]: ...

    def get_latest_profile(self, ticker: str) -> dict[str, Any] | None: ...

    def get_latest_macro_snapshot(self) -> dict[str, Any] | None: ...

    def list_latest_market_indices(self) -> list[dict[str, Any]]: ...

    def get_latest_refresh_status(self) -> dict[str, Any]: ...

    def get_forecast_snapshot(
        self,
        *,
        ticker: str,
        horizon_days: int,
        window_size: int,
        method_version: str = FORECAST_METHOD_VERSION,
    ) -> dict[str, Any] | None: ...

    def list_latest_forecast_snapshots(
        self,
        *,
        horizon_days: int,
        window_size: int,
        method_version: str = FORECAST_METHOD_VERSION,
    ) -> list[dict[str, Any]]: ...

    def deactivate_missing_universe(self, active_tickers: set[str]) -> int: ...


def _today_iso() -> str:
    return date.today().isoformat()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


@dataclass(frozen=True)
class SupabaseMarketDataRepository:
    url: str
    service_role_key: str
    timeout_seconds: float = 30.0

    @property
    def rest_url(self) -> str:
        return f"{self.url.rstrip('/')}/rest/v1"

    def _headers(self, *, prefer: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json",
        }
        if prefer is not None:
            headers["Prefer"] = prefer
        return headers

    def _request(
        self,
        method: str,
        table: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
        prefer: str | None = None,
    ) -> Any:
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.request(
                    method,
                    f"{self.rest_url}/{table}",
                    params=params,
                    json=json_body,
                    headers=self._headers(prefer=prefer),
                )
            response.raise_for_status()
            if response.content:
                return response.json()
            return None
        except httpx.HTTPStatusError as exc:  # pragma: no cover - network/runtime defensive path
            if exc.response.status_code == 404:
                migration = "supabase/migrations/202604240001_market_data.sql"
                if table == "market_index_snapshots":
                    migration = "supabase/migrations/202604250001_market_indices.sql"
                message = (
                    f"Supabase REST could not find table '{table}'. Apply {migration} "
                    "to the public schema and retry after the API schema cache refreshes."
                )
                raise MarketDataUnavailable(message) from exc
            raise MarketDataUnavailable(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - network/runtime defensive path
            raise MarketDataUnavailable(str(exc)) from exc

    def _get(self, table: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        payload = self._request("GET", table, params=params)
        return payload if isinstance(payload, list) else []

    def _get_all(
        self,
        table: str,
        params: dict[str, Any] | None = None,
        *,
        batch_size: int = 1000,
    ) -> list[dict[str, Any]]:
        all_rows: list[dict[str, Any]] = []
        offset = 0
        size = max(int(batch_size), 1)
        while True:
            page_params = dict(params or {})
            page_params["limit"] = str(size)
            page_params["offset"] = str(offset)
            rows = self._get(table, page_params)
            all_rows.extend(rows)
            if len(rows) < size:
                break
            offset += size
        return all_rows

    def health_payload(self) -> dict[str, Any]:
        try:
            rows = self._get("asset_universe", {"select": "ticker", "limit": "1"})
            return {
                "configured": True,
                "status": "ok",
                "source": "supabase",
                "sample_rows": len(rows),
            }
        except MarketDataUnavailable as exc:
            return {
                "configured": True,
                "status": "unavailable",
                "source": "supabase",
                "error": str(exc),
            }

    def validate_schema(self) -> None:
        for table in REQUIRED_SUPABASE_TABLES:
            self._get(table, {"select": "*", "limit": "1"})

    def upsert_rows(
        self,
        table: str,
        rows: list[dict[str, Any]],
        *,
        on_conflict: str,
        batch_size: int = 500,
    ) -> int:
        written = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            if not batch:
                continue
            self._request(
                "POST",
                table,
                params={"on_conflict": on_conflict},
                json_body=batch,
                prefer="resolution=merge-duplicates,return=minimal",
            )
            written += len(batch)
        return written

    def upsert_universe(self, rows: list[dict[str, Any]]) -> int:
        return self.upsert_rows("asset_universe", rows, on_conflict="ticker")

    def deactivate_missing_universe(self, active_tickers: set[str]) -> int:
        normalized = {_normalize_ticker(ticker) for ticker in active_tickers}
        rows = self._get(
            "asset_universe",
            {
                "select": "ticker",
                "active": "eq.true",
                "limit": "5000",
            },
        )
        missing = [
            row["ticker"]
            for row in rows
            if _normalize_ticker(str(row.get("ticker", ""))) not in normalized
        ]
        written = 0
        for start in range(0, len(missing), 100):
            batch = missing[start : start + 100]
            if not batch:
                continue
            quoted = ",".join(f'"{ticker}"' for ticker in batch)
            self._request(
                "PATCH",
                "asset_universe",
                params={"ticker": f"in.({quoted})"},
                json_body={"active": False},
                prefer="return=minimal",
            )
            written += len(batch)
        return written

    def upsert_ohlcv(self, rows: list[dict[str, Any]]) -> int:
        return self.upsert_rows("market_ohlcv_daily", rows, on_conflict="ticker,date")

    def upsert_profiles(self, rows: list[dict[str, Any]]) -> int:
        return self.upsert_rows("asset_profile_snapshots", rows, on_conflict="ticker,as_of_date")

    def upsert_macro(self, rows: list[dict[str, Any]]) -> int:
        return self.upsert_rows("macro_observations", rows, on_conflict="date")

    def upsert_market_indices(self, rows: list[dict[str, Any]]) -> int:
        return self.upsert_rows("market_index_snapshots", rows, on_conflict="symbol,as_of_date")

    def upsert_forecasts(self, rows: list[dict[str, Any]]) -> int:
        return self.upsert_rows(
            "forecast_snapshots",
            rows,
            on_conflict="ticker,as_of_date,horizon_days,window_size,method_version",
        )

    def upsert_refresh_run(self, row: dict[str, Any]) -> int:
        return self.upsert_rows("refresh_runs", [row], on_conflict="id")

    def upsert_refresh_run_items(self, rows: list[dict[str, Any]]) -> int:
        return self.upsert_rows("refresh_run_items", rows, on_conflict="run_id,ticker,stage")

    def list_universe(self) -> list[dict[str, Any]]:
        return self._get(
            "asset_universe",
            {
                "select": "*",
                "active": "eq.true",
                "order": "asset_class.asc,ticker.asc",
            },
        )

    def ticker_exists(self, ticker: str) -> bool:
        return self.ticker_metadata(ticker) is not None

    def ticker_metadata(self, ticker: str) -> dict[str, Any] | None:
        rows = self._get(
            "asset_universe",
            {
                "select": "*",
                "ticker": f"eq.{_normalize_ticker(ticker)}",
                "active": "eq.true",
                "limit": "1",
            },
        )
        return rows[0] if rows else None

    def coverage_for_ticker(self, ticker: str) -> dict[str, Any]:
        normalized = _normalize_ticker(ticker)
        rows = self._get_all(
            "market_ohlcv_daily",
            {
                "select": "date,close",
                "ticker": f"eq.{normalized}",
                "order": "date.asc",
            },
        )
        profile = self.get_latest_profile(normalized)
        if not rows:
            return {
                "first_date": None,
                "latest_date": None,
                "row_count": 0,
                "latest_price": None,
                "has_profile": profile is not None,
                "profile_as_of_date": profile.get("as_of_date") if profile else None,
            }
        return {
            "first_date": rows[0]["date"],
            "latest_date": rows[-1]["date"],
            "row_count": len(rows),
            "latest_price": rows[-1].get("close"),
            "has_profile": profile is not None,
            "profile_as_of_date": profile.get("as_of_date") if profile else None,
        }

    def get_ohlcv_history(self, ticker: str) -> list[dict[str, Any]]:
        return self._get_all(
            "market_ohlcv_daily",
            {
                "select": "*",
                "ticker": f"eq.{_normalize_ticker(ticker)}",
                "order": "date.asc",
            },
        )

    def get_latest_profile(self, ticker: str) -> dict[str, Any] | None:
        rows = self._get(
            "asset_profile_snapshots",
            {
                "select": "*",
                "ticker": f"eq.{_normalize_ticker(ticker)}",
                "order": "as_of_date.desc",
                "limit": "1",
            },
        )
        return rows[0] if rows else None

    def get_latest_macro_snapshot(self) -> dict[str, Any] | None:
        rows = self._get(
            "macro_observations",
            {"select": "*", "order": "date.desc", "limit": "1"},
        )
        return rows[0] if rows else None

    def list_latest_market_indices(self) -> list[dict[str, Any]]:
        rows = self._get(
            "market_index_snapshots",
            {
                "select": "*",
                "order": "as_of_date.desc,display_order.asc",
                "limit": "200",
            },
        )
        latest_by_symbol: dict[str, dict[str, Any]] = {}
        for row in rows:
            latest_by_symbol.setdefault(row["symbol"], row)
        return sorted(
            latest_by_symbol.values(),
            key=lambda row: (int(row.get("display_order") or 0), row.get("symbol") or ""),
        )

    def get_latest_refresh_status(self) -> dict[str, Any]:
        runs = self._get("refresh_runs", {"select": "*", "order": "started_at.desc", "limit": "1"})
        if not runs:
            return {
                "configured": True,
                "source": "supabase",
                "latest_run": None,
                "failed_items": [],
                "stale_tickers": [],
                "latest_market_date": None,
            }
        latest_run = runs[0]
        failed = self._get(
            "refresh_run_items",
            {
                "select": "*",
                "run_id": f"eq.{latest_run['id']}",
                "status": "eq.failed",
                "order": "ticker.asc",
            },
        )
        universe = self.list_universe()
        stale = []
        freshness_cutoff = (date.today() - timedelta(days=10)).isoformat()
        latest_dates = []
        for row in universe:
            coverage = self.coverage_for_ticker(row["ticker"])
            latest = coverage.get("latest_date")
            if latest:
                latest_dates.append(latest)
            if (
                not latest
                or latest < freshness_cutoff
                or coverage.get("row_count", 0) < int(row.get("min_history_days") or 100)
            ):
                stale.append(row["ticker"])
        return {
            "configured": True,
            "source": "supabase",
            "latest_run": latest_run,
            "failed_items": failed,
            "stale_tickers": stale,
            "latest_market_date": max(latest_dates) if latest_dates else None,
        }

    def get_forecast_snapshot(
        self,
        *,
        ticker: str,
        horizon_days: int,
        window_size: int,
        method_version: str = FORECAST_METHOD_VERSION,
    ) -> dict[str, Any] | None:
        rows = self._get(
            "forecast_snapshots",
            {
                "select": "*",
                "ticker": f"eq.{_normalize_ticker(ticker)}",
                "horizon_days": f"eq.{int(horizon_days)}",
                "window_size": f"eq.{int(window_size)}",
                "method_version": f"eq.{method_version}",
                "order": "as_of_date.desc",
                "limit": "1",
            },
        )
        return rows[0] if rows else None

    def list_latest_forecast_snapshots(
        self,
        *,
        horizon_days: int,
        window_size: int,
        method_version: str = FORECAST_METHOD_VERSION,
    ) -> list[dict[str, Any]]:
        rows = self._get(
            "forecast_snapshots",
            {
                "select": "*",
                "horizon_days": f"eq.{int(horizon_days)}",
                "window_size": f"eq.{int(window_size)}",
                "method_version": f"eq.{method_version}",
                "order": "as_of_date.desc",
                "limit": "2000",
            },
        )
        latest_by_ticker: dict[str, dict[str, Any]] = {}
        for row in rows:
            ticker = row["ticker"]
            if ticker not in latest_by_ticker:
                latest_by_ticker[ticker] = row
        return list(latest_by_ticker.values())


class InMemoryMarketDataRepository:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {
            "asset_universe": [],
            "market_ohlcv_daily": [],
            "asset_profile_snapshots": [],
            "macro_observations": [],
            "market_index_snapshots": [],
            "forecast_snapshots": [],
            "refresh_runs": [],
            "refresh_run_items": [],
        }

    def health_payload(self) -> dict[str, Any]:
        return {
            "configured": True,
            "status": "ok",
            "source": "memory",
            "sample_rows": len(self.tables["asset_universe"]),
        }

    def _upsert(self, table: str, rows: list[dict[str, Any]], keys: tuple[str, ...]) -> int:
        target = self.tables[table]
        for row in rows:
            normalized = deepcopy(row)
            match = next(
                (
                    existing
                    for existing in target
                    if all(existing.get(key) == normalized.get(key) for key in keys)
                ),
                None,
            )
            if match is None:
                target.append(normalized)
            else:
                match.update(normalized)
        return len(rows)

    def upsert_universe(self, rows: list[dict[str, Any]]) -> int:
        return self._upsert("asset_universe", rows, ("ticker",))

    def deactivate_missing_universe(self, active_tickers: set[str]) -> int:
        normalized = {_normalize_ticker(ticker) for ticker in active_tickers}
        written = 0
        for row in self.tables["asset_universe"]:
            ticker = _normalize_ticker(str(row.get("ticker", "")))
            if row.get("active", True) and ticker not in normalized:
                row["active"] = False
                written += 1
        return written

    def upsert_ohlcv(self, rows: list[dict[str, Any]]) -> int:
        return self._upsert("market_ohlcv_daily", rows, ("ticker", "date"))

    def upsert_profiles(self, rows: list[dict[str, Any]]) -> int:
        return self._upsert("asset_profile_snapshots", rows, ("ticker", "as_of_date"))

    def upsert_macro(self, rows: list[dict[str, Any]]) -> int:
        return self._upsert("macro_observations", rows, ("date",))

    def upsert_market_indices(self, rows: list[dict[str, Any]]) -> int:
        return self._upsert("market_index_snapshots", rows, ("symbol", "as_of_date"))

    def upsert_forecasts(self, rows: list[dict[str, Any]]) -> int:
        return self._upsert(
            "forecast_snapshots",
            rows,
            ("ticker", "as_of_date", "horizon_days", "window_size", "method_version"),
        )

    def upsert_refresh_run(self, row: dict[str, Any]) -> int:
        row = {**row, "id": row.get("id") or str(uuid.uuid4())}
        return self._upsert("refresh_runs", [row], ("id",))

    def upsert_refresh_run_items(self, rows: list[dict[str, Any]]) -> int:
        return self._upsert("refresh_run_items", rows, ("run_id", "ticker", "stage"))

    def list_universe(self) -> list[dict[str, Any]]:
        rows = [row for row in self.tables["asset_universe"] if row.get("active", True)]
        return sorted(rows, key=lambda row: (row.get("asset_class", ""), row.get("ticker", "")))

    def ticker_exists(self, ticker: str) -> bool:
        return self.ticker_metadata(ticker) is not None

    def ticker_metadata(self, ticker: str) -> dict[str, Any] | None:
        normalized = _normalize_ticker(ticker)
        return next(
            (
                row
                for row in self.tables["asset_universe"]
                if row.get("ticker") == normalized and row.get("active", True)
            ),
            None,
        )

    def get_ohlcv_history(self, ticker: str) -> list[dict[str, Any]]:
        normalized = _normalize_ticker(ticker)
        rows = [
            row
            for row in self.tables["market_ohlcv_daily"]
            if row.get("ticker") == normalized
        ]
        return sorted(rows, key=lambda row: row["date"])

    def get_latest_profile(self, ticker: str) -> dict[str, Any] | None:
        normalized = _normalize_ticker(ticker)
        rows = [
            row
            for row in self.tables["asset_profile_snapshots"]
            if row.get("ticker") == normalized
        ]
        if not rows:
            return None
        return sorted(rows, key=lambda row: row["as_of_date"])[-1]

    def coverage_for_ticker(self, ticker: str) -> dict[str, Any]:
        rows = self.get_ohlcv_history(ticker)
        profile = self.get_latest_profile(ticker)
        if not rows:
            return {
                "first_date": None,
                "latest_date": None,
                "row_count": 0,
                "latest_price": None,
                "has_profile": profile is not None,
                "profile_as_of_date": profile.get("as_of_date") if profile else None,
            }
        return {
            "first_date": rows[0]["date"],
            "latest_date": rows[-1]["date"],
            "row_count": len(rows),
            "latest_price": rows[-1].get("close"),
            "has_profile": profile is not None,
            "profile_as_of_date": profile.get("as_of_date") if profile else None,
        }

    def get_latest_macro_snapshot(self) -> dict[str, Any] | None:
        rows = self.tables["macro_observations"]
        if not rows:
            return None
        return sorted(rows, key=lambda row: row["date"])[-1]

    def list_latest_market_indices(self) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in sorted(
            self.tables["market_index_snapshots"],
            key=lambda item: item["as_of_date"],
            reverse=True,
        ):
            latest.setdefault(row["symbol"], row)
        return sorted(
            latest.values(),
            key=lambda row: (int(row.get("display_order") or 0), row.get("symbol") or ""),
        )

    def get_latest_refresh_status(self) -> dict[str, Any]:
        runs = sorted(self.tables["refresh_runs"], key=lambda row: row["started_at"])
        latest_run = runs[-1] if runs else None
        latest_dates = [
            coverage["latest_date"]
            for coverage in (self.coverage_for_ticker(row["ticker"]) for row in self.list_universe())
            if coverage["latest_date"]
        ]
        failed_items = []
        if latest_run:
            failed_items = [
                row
                for row in self.tables["refresh_run_items"]
                if row.get("run_id") == latest_run.get("id") and row.get("status") == "failed"
            ]
        stale = []
        freshness_cutoff = (date.today() - timedelta(days=10)).isoformat()
        for row in self.list_universe():
            coverage = self.coverage_for_ticker(row["ticker"])
            latest = coverage.get("latest_date")
            if (
                not latest
                or latest < freshness_cutoff
                or coverage["row_count"] < int(row.get("min_history_days") or 100)
            ):
                stale.append(row["ticker"])
        return {
            "configured": True,
            "source": "memory",
            "latest_run": latest_run,
            "failed_items": failed_items,
            "stale_tickers": stale,
            "latest_market_date": max(latest_dates) if latest_dates else None,
        }

    def get_forecast_snapshot(
        self,
        *,
        ticker: str,
        horizon_days: int,
        window_size: int,
        method_version: str = FORECAST_METHOD_VERSION,
    ) -> dict[str, Any] | None:
        normalized = _normalize_ticker(ticker)
        rows = [
            row
            for row in self.tables["forecast_snapshots"]
            if row.get("ticker") == normalized
            and int(row.get("horizon_days")) == int(horizon_days)
            and int(row.get("window_size")) == int(window_size)
            and row.get("method_version") == method_version
        ]
        return sorted(rows, key=lambda row: row["as_of_date"])[-1] if rows else None

    def list_latest_forecast_snapshots(
        self,
        *,
        horizon_days: int,
        window_size: int,
        method_version: str = FORECAST_METHOD_VERSION,
    ) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self.tables["forecast_snapshots"]
            if int(row.get("horizon_days")) == int(horizon_days)
            and int(row.get("window_size")) == int(window_size)
            and row.get("method_version") == method_version
        ]
        latest: dict[str, dict[str, Any]] = {}
        for row in sorted(rows, key=lambda item: item["as_of_date"], reverse=True):
            latest.setdefault(row["ticker"], row)
        return list(latest.values())


def build_market_repository(settings: Settings) -> MarketDataRepository | None:
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return None
    if settings.supabase_url.startswith("memory://"):
        return InMemoryMarketDataRepository()
    return SupabaseMarketDataRepository(
        url=settings.supabase_url,
        service_role_key=settings.supabase_service_role_key,
    )


def empty_refresh_status() -> dict[str, Any]:
    return {
        "configured": False,
        "source": "local_artifacts",
        "latest_run": None,
        "failed_items": [],
        "stale_tickers": [],
        "latest_market_date": None,
        "message": "Supabase market data is not configured.",
        "checked_at": _now_iso(),
        "today": _today_iso(),
    }
