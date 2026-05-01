from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Volume")
MACRO_FEATURE_COLUMNS = (
    "VIX Market Volatility",
    "Federal Funds Rate",
    "10-Year Treasury Yield",
    "Unemployment Rate",
    "CPI All Items",
    "Recession Indicator",
)


class MarketDataProvider(Protocol):
    name: str

    def fetch_daily_ohlcv(
        self,
        tickers: list[dict[str, Any]],
        *,
        start_date: str,
        end_date: str,
        batch_size: int = 10,
        freshness_days: int = 10,
    ) -> list[dict[str, Any]]: ...

    def fetch_company_profiles(self, tickers: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def fetch_macro_observations(
        self,
        *,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]: ...

    def fetch_market_indices(
        self,
        indices: list[dict[str, Any]],
        *,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]: ...


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


def _clean_date(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=UTC).date().isoformat()
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date().isoformat()
    except Exception:
        return None


def _ensure_multiindex(frame: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        if len(symbols) != 1:
            raise ValueError("Expected multi-index columns for multiple tickers")
        return pd.concat({symbols[0]: frame}, axis=1)
    level0 = set(frame.columns.get_level_values(0))
    level1 = set(frame.columns.get_level_values(1))
    if level0.intersection(set(OHLCV_FIELDS)) or "Adj Close" in level0:
        if level1.intersection(set(symbols)):
            frame = frame.swaplevel(axis=1)
    return frame.sort_index(axis=1)


def _expected_latest_cutoff(end_date: str, freshness_days: int) -> date:
    return date.fromisoformat(end_date) - timedelta(days=max(int(freshness_days), 1))


@dataclass(frozen=True)
class YFinanceProvider:
    macro_path: Path | None = None
    name: str = "yfinance"

    def _download_rows(
        self,
        tickers: list[dict[str, Any]],
        *,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        import yfinance as yf

        if not tickers:
            return []
        symbols = [row["provider_symbol"] for row in tickers]
        symbol_to_ticker = {row["provider_symbol"]: row["ticker"] for row in tickers}
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
            raise ValueError(f"No market data returned for {len(symbols)} tickers")
        normalized = _ensure_multiindex(frame, symbols)
        rows: list[dict[str, Any]] = []
        ingested_at = datetime.now(UTC).isoformat()
        for symbol in symbols:
            if symbol not in normalized.columns.get_level_values(0):
                continue
            ticker_frame = normalized[symbol].copy()
            if "Close" not in ticker_frame.columns:
                continue
            for timestamp, values in ticker_frame.iterrows():
                close = _clean_number(values.get("Close"))
                if close is None or close <= 0:
                    continue
                rows.append(
                    {
                        "ticker": symbol_to_ticker[symbol],
                        "date": pd.Timestamp(timestamp).date().isoformat(),
                        "open": _clean_number(values.get("Open")),
                        "high": _clean_number(values.get("High")),
                        "low": _clean_number(values.get("Low")),
                        "close": close,
                        "adjusted_close": _clean_number(values.get("Adj Close")) or close,
                        "volume": _clean_number(values.get("Volume")),
                        "provider": self.name,
                        "ingested_at": ingested_at,
                    }
                )
        return rows

    def fetch_daily_ohlcv(
        self,
        tickers: list[dict[str, Any]],
        *,
        start_date: str,
        end_date: str,
        batch_size: int = 10,
        freshness_days: int = 10,
    ) -> list[dict[str, Any]]:
        all_rows: list[dict[str, Any]] = []
        for start in range(0, len(tickers), max(int(batch_size), 1)):
            batch = tickers[start : start + max(int(batch_size), 1)]
            all_rows.extend(
                self._download_rows(
                    batch,
                    start_date=start_date,
                    end_date=end_date,
                )
            )

        cutoff = _expected_latest_cutoff(end_date, freshness_days)
        latest_by_ticker: dict[str, date] = {}
        for row in all_rows:
            row_date = date.fromisoformat(row["date"])
            ticker = row["ticker"]
            latest_by_ticker[ticker] = max(latest_by_ticker.get(ticker, row_date), row_date)

        retry_assets = [
            asset
            for asset in tickers
            if latest_by_ticker.get(asset["ticker"]) is None
            or latest_by_ticker[asset["ticker"]] < cutoff
        ]
        if retry_assets:
            stale_tickers = {asset["ticker"] for asset in retry_assets}
            all_rows = [row for row in all_rows if row["ticker"] not in stale_tickers]
            for asset in retry_assets:
                all_rows.extend(
                    self._download_rows(
                        [asset],
                        start_date=asset.get("start_date") or start_date,
                        end_date=end_date,
                    )
                )
        return all_rows

    def fetch_company_profiles(self, tickers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        import yfinance as yf

        rows = []
        as_of_date = date.today().isoformat()
        ingested_at = datetime.now(UTC).isoformat()
        for entry in tickers:
            symbol = entry["provider_symbol"]
            ticker = yf.Ticker(symbol)
            info: dict[str, Any] = {}
            try:
                info = ticker.info or {}
            except Exception:
                info = {}
            fast_info: dict[str, Any] = {}
            try:
                fast_info_obj = ticker.fast_info
                fast_info = dict(fast_info_obj.items()) if hasattr(fast_info_obj, "items") else {}
            except Exception:
                fast_info = {}

            last_sale = (
                _clean_number(fast_info.get("last_price"))
                or _clean_number(info.get("regularMarketPrice"))
                or _clean_number(info.get("currentPrice"))
            )
            rows.append(
                {
                    "ticker": entry["ticker"],
                    "as_of_date": as_of_date,
                    "market_cap": _clean_number(info.get("marketCap")),
                    "pe_ratio": _clean_number(info.get("trailingPE")),
                    "fifty_two_week_high": _clean_number(info.get("fiftyTwoWeekHigh"))
                    or _clean_number(fast_info.get("year_high")),
                    "fifty_two_week_low": _clean_number(info.get("fiftyTwoWeekLow"))
                    or _clean_number(fast_info.get("year_low")),
                    "average_volume": _clean_number(info.get("averageVolume"))
                    or _clean_number(fast_info.get("three_month_average_volume")),
                    "volume": _clean_number(info.get("volume")),
                    "dividend_yield": _clean_number(info.get("dividendYield")),
                    "dividend_frequency": None,
                    "ex_dividend_date": _clean_date(info.get("exDividendDate")),
                    "bid": _clean_number(info.get("bid")),
                    "ask": _clean_number(info.get("ask")),
                    "last_sale": last_sale,
                    "day_open": _clean_number(info.get("regularMarketOpen"))
                    or _clean_number(fast_info.get("open")),
                    "day_high": _clean_number(info.get("dayHigh"))
                    or _clean_number(fast_info.get("day_high")),
                    "day_low": _clean_number(info.get("dayLow"))
                    or _clean_number(fast_info.get("day_low")),
                    "exchange": info.get("exchange") or entry.get("exchange"),
                    "margin_requirement": None,
                    "raw_payload": {
                        "quoteType": info.get("quoteType"),
                        "shortName": info.get("shortName"),
                        "longName": info.get("longName"),
                        "sector": info.get("sector"),
                        "industry": info.get("industry"),
                    },
                    "ingested_at": ingested_at,
                }
            )
        return rows

    def fetch_macro_observations(
        self,
        *,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        if self.macro_path is None or not self.macro_path.exists():
            return []
        macro = pd.read_csv(self.macro_path, parse_dates=["Date"])
        required = ["Date", *MACRO_FEATURE_COLUMNS]
        missing = [column for column in required if column not in macro.columns]
        if missing:
            raise ValueError(f"Macro dataset missing columns: {missing}")
        macro = macro.loc[:, required].sort_values("Date")
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        macro = macro[(macro["Date"] >= start) & (macro["Date"] < end)]
        ingested_at = datetime.now(UTC).isoformat()
        rows = []
        for _, row in macro.iterrows():
            rows.append(
                {
                    "date": pd.Timestamp(row["Date"]).date().isoformat(),
                    "vix": _clean_number(row["VIX Market Volatility"]),
                    "federal_funds_rate": _clean_number(row["Federal Funds Rate"]),
                    "treasury_10y": _clean_number(row["10-Year Treasury Yield"]),
                    "unemployment_rate": _clean_number(row["Unemployment Rate"]),
                    "cpi_all_items": _clean_number(row["CPI All Items"]),
                    "recession_indicator": _clean_number(row["Recession Indicator"]),
                    "provider": "macro_csv",
                    "ingested_at": ingested_at,
                }
            )
        return rows

    def fetch_market_indices(
        self,
        indices: list[dict[str, Any]],
        *,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        import yfinance as yf

        if not indices:
            return []
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
        ingested_at = datetime.now(UTC).isoformat()
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
                    "provider": self.name,
                    "display_order": int(metadata.get("display_order") or 0),
                    "raw_payload": {"provider_symbol": provider_symbol},
                    "ingested_at": ingested_at,
                }
            )
        return rows


def build_provider(name: str, *, macro_path: Path | None = None) -> MarketDataProvider:
    provider_name = name.strip().lower()
    if provider_name == "yfinance":
        return YFinanceProvider(macro_path=macro_path)
    raise ValueError(f"Unsupported market data provider: {name}")
