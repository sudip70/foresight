from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import Settings


OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Volume")


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
                "currency": row.get("currency") or "USD",
                "display_order": int(row.get("display_order") or 0),
            }
        )
    return normalized


def refresh_market_index_snapshots(
    settings: Settings,
    *,
    repository=None,
) -> dict[str, Any]:
    if not settings.market_index_auto_refresh:
        return {"enabled": False, "rows": [], "rows_written": 0}

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
