from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import Settings


OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Volume")
HISTORY_RANGE_LOOKBACK_DAYS = {
    "1m": 45,
    "3m": 110,
    "6m": 220,
    "1y": 400,
    "5y": 365 * 5 + 30,
}


def _clean_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        parsed = float(value)
        if np.isfinite(parsed):
            return parsed
    except (TypeError, ValueError):
        return None
    return None


def _ensure_multiindex(frame: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        if len(symbols) != 1:
            raise ValueError("Expected multi-index columns for multiple index symbols")
        return pd.concat({symbols[0]: frame}, axis=1)
    level0 = set(frame.columns.get_level_values(0))
    level1 = set(frame.columns.get_level_values(1))
    if level0.intersection(set(OHLCV_FIELDS)) or "Adj Close" in level0:
        return frame.swaplevel(axis=1).sort_index(axis=1)
    if level1.intersection(set(OHLCV_FIELDS)) or "Adj Close" in level1:
        return frame.sort_index(axis=1)
    return frame


def _fetch_yfinance_market_indices(
    indices: list[dict[str, Any]],
    *,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    import yfinance as yf

    symbols = [row["provider_symbol"] for row in indices]
    by_symbol = {row["provider_symbol"]: row for row in indices}
    frame = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if frame.empty:
        raise ValueError(f"No index data returned for {len(symbols)} indices")

    normalized = _ensure_multiindex(frame, symbols)
    rows: list[dict[str, Any]] = []
    for provider_symbol in symbols:
        if provider_symbol not in normalized.columns.get_level_values(0):
            continue
        metadata = by_symbol[provider_symbol]
        ticker_frame = normalized[provider_symbol].copy()
        if "Close" not in ticker_frame.columns:
            continue
        clean = ticker_frame[
            ticker_frame["Close"].map(lambda value: (_clean_number(value) or 0.0) > 0)
        ].sort_index()
        if clean.empty:
            continue

        latest = clean.iloc[-1]
        previous = clean.iloc[-2] if len(clean) > 1 else latest
        value = _clean_number(latest.get("Close"))
        previous_close = _clean_number(previous.get("Close"))
        if value is None:
            continue
        change = None
        change_percent = None
        if previous_close is not None and previous_close > 0:
            change = value - previous_close
            change_percent = change / previous_close

        rows.append(
            {
                "symbol": metadata["symbol"],
                "as_of_date": pd.Timestamp(clean.index[-1]).date().isoformat(),
                "label": metadata.get("label") or metadata["symbol"],
                "display_name": metadata.get("display_name") or metadata.get("label"),
                "provider_symbol": provider_symbol,
                "value": value,
                "previous_close": previous_close,
                "change": change,
                "change_percent": change_percent,
                "day_open": _clean_number(latest.get("Open")),
                "day_high": _clean_number(latest.get("High")),
                "day_low": _clean_number(latest.get("Low")),
                "volume": _clean_number(latest.get("Volume")),
                "currency": metadata.get("currency") or "USD",
                "provider": "yfinance",
                "display_order": int(metadata.get("display_order") or 0),
                "raw_payload": {"provider_symbol": provider_symbol},
            }
        )
    return rows


def _normalize_single_symbol_frame(frame: pd.DataFrame, provider_symbol: str) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame.copy()
    normalized = _ensure_multiindex(frame, [provider_symbol])
    if provider_symbol not in normalized.columns.get_level_values(0):
        raise ValueError(f"No index history returned for {provider_symbol}")
    return normalized[provider_symbol].copy()


def _fetch_yfinance_market_index_history(
    index: dict[str, Any],
    *,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    import yfinance as yf

    provider_symbol = index["provider_symbol"]
    frame = yf.download(
        provider_symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError(f"No index history returned for {provider_symbol}")

    normalized = _normalize_single_symbol_frame(frame, provider_symbol)
    if "Close" not in normalized.columns:
        raise ValueError(f"No close-price history returned for {provider_symbol}")

    rows: list[dict[str, Any]] = []
    for timestamp, row in normalized.sort_index().iterrows():
        close = _clean_number(row.get("Close"))
        if close is None or close <= 0:
            continue
        rows.append(
            {
                "date": pd.Timestamp(timestamp).date().isoformat(),
                "open": _clean_number(row.get("Open")),
                "high": _clean_number(row.get("High")),
                "low": _clean_number(row.get("Low")),
                "close": close,
                "adjusted_close": _clean_number(row.get("Adj Close")),
                "volume": _clean_number(row.get("Volume")),
            }
        )
    if not rows:
        raise ValueError(f"No usable index history returned for {provider_symbol}")

    previous_close = None
    for row in rows:
        close = row["close"]
        change = None if previous_close is None else close - previous_close
        change_percent = (
            None
            if previous_close is None or previous_close <= 0
            else change / previous_close
        )
        row["change"] = change
        row["change_percent"] = change_percent
        previous_close = close
    return rows


def load_market_index_config(path: Path) -> list[dict[str, Any]]:
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
                "fallback_ticker": row.get("fallback_ticker"),
                "currency": row.get("currency") or "USD",
                "display_order": int(row.get("display_order") or 0),
            }
        )
    return normalized


def _repository_index_history_rows(
    repository,
    *,
    fallback_ticker: str,
    lookback_days: int,
) -> list[dict[str, Any]]:
    rows = repository.get_ohlcv_history(fallback_ticker)
    if not rows:
        return []
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    filtered = [row for row in rows if str(row.get("date")) >= cutoff]
    if len(filtered) < 2:
        filtered = rows[-min(len(rows), max(lookback_days, 2)) :]

    history: list[dict[str, Any]] = []
    previous_close = None
    for row in filtered:
        close = _clean_number(row.get("close"))
        if close is None or close <= 0:
            continue
        change = None if previous_close is None else close - previous_close
        change_percent = (
            None
            if previous_close is None or previous_close <= 0
            else change / previous_close
        )
        history.append(
            {
                "date": row.get("date"),
                "open": _clean_number(row.get("open")),
                "high": _clean_number(row.get("high")),
                "low": _clean_number(row.get("low")),
                "close": close,
                "adjusted_close": _clean_number(row.get("adjusted_close")),
                "volume": _clean_number(row.get("volume")),
                "change": change,
                "change_percent": change_percent,
            }
        )
        previous_close = close
    return history


def _history_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    latest = history[-1]
    previous = history[-2] if len(history) > 1 else latest
    first_close = float(history[0]["close"])
    latest_close = float(latest["close"])
    closes = [float(row["close"]) for row in history]
    return {
        "first_date": history[0]["date"],
        "latest_date": latest["date"],
        "first_close": first_close,
        "latest_close": latest_close,
        "previous_close": previous["close"],
        "change": None if len(history) <= 1 else latest_close - float(previous["close"]),
        "change_percent": (
            None
            if len(history) <= 1 or float(previous["close"]) <= 0
            else (latest_close - float(previous["close"])) / float(previous["close"])
        ),
        "range_return": (
            None if first_close <= 0 else (latest_close - first_close) / first_close
        ),
        "high": max(closes),
        "low": min(closes),
        "points": len(history),
    }


def fetch_market_index_history_from_repository(
    settings: Settings,
    *,
    repository,
    symbol: str,
    history_range: str = "1y",
) -> dict[str, Any]:
    normalized_range = history_range.strip().lower()
    lookback_days = HISTORY_RANGE_LOOKBACK_DAYS.get(normalized_range)
    if lookback_days is None:
        supported = ", ".join(sorted(HISTORY_RANGE_LOOKBACK_DAYS))
        raise ValueError(
            f"Unsupported market index history range: {history_range}. Use {supported}."
        )

    normalized_symbol = symbol.strip().upper()
    indices = load_market_index_config(settings.market_index_config_path)
    index = next((row for row in indices if row["symbol"] == normalized_symbol), None)
    if index is None:
        raise ValueError(f"Unsupported market index symbol: {normalized_symbol}")
    fallback_ticker = index.get("fallback_ticker")
    if not fallback_ticker:
        raise ValueError(f"No Supabase proxy history configured for {normalized_symbol}")

    history = _repository_index_history_rows(
        repository,
        fallback_ticker=str(fallback_ticker),
        lookback_days=lookback_days,
    )
    if not history:
        raise ValueError(f"No Supabase proxy history returned for {normalized_symbol}")
    summary = _history_summary(history)
    return {
        "source": "supabase_proxy",
        "symbol": index["symbol"],
        "label": index["label"],
        "display_name": f"{index['display_name']} proxy",
        "provider_symbol": str(fallback_ticker),
        "currency": index["currency"],
        "range": normalized_range,
        "lookback_days": lookback_days,
        "as_of_date": summary["latest_date"],
        "history": history,
        "summary": summary,
        "disclaimer": (
            f"Historical index chart uses {fallback_ticker} Supabase OHLCV as a proxy "
            "when live index-provider history is unavailable."
        ),
    }


def fetch_market_index_snapshots_from_repository(
    settings: Settings,
    *,
    repository,
) -> dict[str, Any]:
    indices = load_market_index_config(settings.market_index_config_path)
    rows: list[dict[str, Any]] = []
    for index in indices:
        fallback_ticker = index.get("fallback_ticker")
        if not fallback_ticker:
            continue
        history = _repository_index_history_rows(
            repository,
            fallback_ticker=str(fallback_ticker),
            lookback_days=max(settings.market_index_refresh_lookback_days, 2),
        )
        if not history:
            continue
        latest = history[-1]
        previous = history[-2] if len(history) > 1 else latest
        value = float(latest["close"])
        previous_close = float(previous["close"])
        change = None if len(history) <= 1 else value - previous_close
        change_percent = (
            None if len(history) <= 1 or previous_close <= 0 else change / previous_close
        )
        rows.append(
            {
                "symbol": index["symbol"],
                "as_of_date": latest["date"],
                "label": index["label"],
                "display_name": f"{index['display_name']} proxy",
                "provider_symbol": str(fallback_ticker),
                "value": value,
                "previous_close": previous_close,
                "change": change,
                "change_percent": change_percent,
                "day_open": latest.get("open"),
                "day_high": latest.get("high"),
                "day_low": latest.get("low"),
                "volume": latest.get("volume"),
                "currency": index["currency"],
                "provider": "supabase_proxy",
                "display_order": index["display_order"],
                "raw_payload": {"fallback_ticker": fallback_ticker},
            }
        )
    as_of_dates = [str(row["as_of_date"]) for row in rows if row.get("as_of_date")]
    return {
        "enabled": True,
        "provider": "supabase_proxy",
        "as_of_date": max(as_of_dates) if as_of_dates else None,
        "rows": rows,
        "rows_written": 0,
    }


def fetch_market_index_history(
    settings: Settings,
    *,
    symbol: str,
    history_range: str = "1y",
) -> dict[str, Any]:
    normalized_range = history_range.strip().lower()
    lookback_days = HISTORY_RANGE_LOOKBACK_DAYS.get(normalized_range)
    if lookback_days is None:
        supported = ", ".join(sorted(HISTORY_RANGE_LOOKBACK_DAYS))
        raise ValueError(
            f"Unsupported market index history range: {history_range}. Use {supported}."
        )

    normalized_symbol = symbol.strip().upper()
    indices = load_market_index_config(settings.market_index_config_path)
    index = next((row for row in indices if row["symbol"] == normalized_symbol), None)
    if index is None:
        raise ValueError(f"Unsupported market index symbol: {normalized_symbol}")

    provider_name = settings.market_data_provider.strip().lower()
    if provider_name != "yfinance":
        raise ValueError(f"Unsupported market index provider: {settings.market_data_provider}")

    end_date = (date.today() + timedelta(days=1)).isoformat()
    start_date = (date.today() - timedelta(days=lookback_days)).isoformat()
    history = _fetch_yfinance_market_index_history(
        index,
        start_date=start_date,
        end_date=end_date,
    )
    latest = history[-1]
    previous = history[-2] if len(history) > 1 else latest
    first_close = history[0]["close"]
    latest_close = latest["close"]
    range_return = None
    if first_close > 0:
        range_return = (latest_close - first_close) / first_close

    closes = [row["close"] for row in history]
    summary = {
        "first_date": history[0]["date"],
        "latest_date": latest["date"],
        "first_close": first_close,
        "latest_close": latest_close,
        "previous_close": previous["close"],
        "change": None if len(history) <= 1 else latest_close - previous["close"],
        "change_percent": (
            None
            if len(history) <= 1 or previous["close"] <= 0
            else (latest_close - previous["close"]) / previous["close"]
        ),
        "range_return": range_return,
        "high": max(closes),
        "low": min(closes),
        "points": len(history),
    }

    return {
        "source": provider_name,
        "symbol": index["symbol"],
        "label": index["label"],
        "display_name": index["display_name"],
        "provider_symbol": index["provider_symbol"],
        "currency": index["currency"],
        "range": normalized_range,
        "lookback_days": lookback_days,
        "as_of_date": latest["date"],
        "history": history,
        "summary": summary,
        "disclaimer": (
            "Historical index levels are fetched from the configured market data provider "
            "on request."
        ),
    }


def refresh_market_index_snapshots(
    settings: Settings,
    *,
    repository=None,
) -> dict[str, Any]:
    if not settings.market_index_auto_refresh:
        return {"enabled": False, "rows": [], "rows_written": 0}

    return fetch_market_index_snapshots(settings, repository=repository)


def fetch_market_index_snapshots(
    settings: Settings,
    *,
    repository=None,
) -> dict[str, Any]:
    """Fetch the latest configured index levels without requiring startup refresh."""

    indices = load_market_index_config(settings.market_index_config_path)
    if not indices:
        return {"enabled": True, "rows": [], "rows_written": 0}

    end_date = (date.today() + timedelta(days=1)).isoformat()
    start_date = (
        date.today() - timedelta(days=max(settings.market_index_refresh_lookback_days, 1))
    ).isoformat()
    provider_name = settings.market_data_provider.strip().lower()
    if provider_name != "yfinance":
        raise ValueError(f"Unsupported market index provider: {settings.market_data_provider}")
    rows = _fetch_yfinance_market_indices(indices, start_date=start_date, end_date=end_date)

    rows_written = 0
    if repository is not None and hasattr(repository, "upsert_market_indices"):
        rows_written = repository.upsert_market_indices(rows)

    as_of_dates = [str(row["as_of_date"]) for row in rows if row.get("as_of_date")]
    return {
        "enabled": True,
        "provider": provider_name,
        "start_date": start_date,
        "end_date": end_date,
        "as_of_date": max(as_of_dates) if as_of_dates else None,
        "rows": rows,
        "rows_written": rows_written,
    }
