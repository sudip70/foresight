from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
import json
import math
from typing import Any

import numpy as np

from backend.app.core.config import Settings
from backend.app.market.repository import (
    FORECAST_METHOD_VERSION,
    MarketDataRepository,
)
from backend.app.ml.artifacts import ArtifactValidationError


CASH_ASSET_CLASS = "cash"
CASH_TICKER = "CASH"


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str) and not value.strip():
            return default
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except (TypeError, ValueError):
        pass
    return default


def _date_add_days(value: str, days: int) -> str:
    return (date.fromisoformat(value) + timedelta(days=int(days))).isoformat()


def _forecast_start_date() -> str:
    return date.today().isoformat()


def _rebase_forecast_paths(
    forecast_paths: dict[str, Any],
    forecast_start_date: str,
) -> dict[str, list[dict[str, Any]]]:
    rebased: dict[str, list[dict[str, Any]]] = {}
    for scenario, path in forecast_paths.items():
        if not isinstance(path, list):
            continue
        scenario_path = []
        for index, point in enumerate(path):
            if not isinstance(point, dict):
                continue
            day = int(point.get("day", index))
            scenario_path.append(
                {
                    **point,
                    "day": day,
                    "date": _date_add_days(forecast_start_date, day),
                }
            )
        rebased[str(scenario)] = scenario_path
    return rebased


def _historical_drawdown(prices: np.ndarray) -> float:
    series = np.asarray(prices, dtype=float).reshape(-1)
    if series.shape[0] <= 1:
        return 0.0
    running_peak = np.maximum.accumulate(series)
    drawdown = 1.0 - (series / np.clip(running_peak, 1e-12, None))
    return float(np.max(drawdown))


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.70:
        return "High"
    if confidence >= 0.45:
        return "Medium"
    return "Low"


def _risk_label(annualized_volatility: float, max_drawdown: float) -> str:
    risk_score = annualized_volatility + (0.60 * max_drawdown)
    if risk_score >= 0.65:
        return "High"
    if risk_score >= 0.30:
        return "Moderate"
    return "Lower"


def _return_caps(asset_class: str) -> tuple[float, float]:
    if asset_class == "crypto":
        return -0.0030, 0.00125
    if asset_class == "stock":
        return -0.0016, 0.00075
    return -0.0012, 0.00055


def _soft_cap_return(value: float, *, lower: float, upper: float) -> float:
    if value >= 0.0:
        return float(upper * math.tanh(value / max(upper, 1e-12)))
    lower_abs = abs(lower)
    return float(-lower_abs * math.tanh(abs(value) / max(lower_abs, 1e-12)))


def _normalize_weights_with_caps(scores: np.ndarray, cap: float) -> np.ndarray:
    weights = np.asarray(scores, dtype=float)
    weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights, dtype=float)
    weights = weights / float(weights.sum())
    cap_value = max(float(cap), 1e-9)
    for _ in range(20):
        over = weights > cap_value
        if not bool(np.any(over)):
            break
        excess = float(np.sum(weights[over] - cap_value))
        weights[over] = cap_value
        under = ~over
        under_sum = float(np.sum(weights[under]))
        if under_sum <= 1e-12:
            break
        weights[under] += excess * (weights[under] / under_sum)
    total = float(weights.sum())
    if total > 1e-12:
        weights = weights / total
    return weights


class SupabaseForecastEngine:
    def __init__(self, repository: MarketDataRepository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings
        self._market_forecast_cache: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
        self._macro_payload_cache: dict[str, Any] | None = None
        self._universe_payload_cache: dict[str, Any] | None = None

    def _cash_daily_return(self) -> float:
        return float((1.0 + self.settings.meta_cash_annual_return) ** (1.0 / 252.0) - 1.0)

    def _forecast_score(self, forecast: dict[str, Any], *, risk: float) -> float:
        risk_value = float(np.clip(risk, 0.0, 1.0))
        base_return = float(forecast["returns"]["base"])
        bull_return = float(forecast["returns"]["bull"])
        bear_return = float(forecast["returns"]["bear"])
        volatility = float(forecast["risk_metrics"]["annualized_volatility"])
        confidence = float(forecast["confidence"])
        downside = max(-bear_return, 0.0)
        return float(
            base_return
            + (risk_value * 0.35 * max(bull_return, 0.0))
            - ((1.0 - risk_value) * 0.65 * downside)
            - ((0.15 + (0.20 * (1.0 - risk_value))) * volatility)
            + (0.12 * confidence)
        )

    def _ticker_rows(self, ticker: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        normalized = _normalize_ticker(ticker)
        metadata = self.repository.ticker_metadata(normalized)
        if metadata is None:
            raise ArtifactValidationError(f"Unsupported ticker: {ticker}")
        rows = self.repository.get_ohlcv_history(normalized)
        if len(rows) < 30:
            raise ArtifactValidationError(
                f"Not enough Supabase market rows for {normalized}: {len(rows)}"
            )
        return metadata, rows

    def _prepare_price_series(self, rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
        clean_rows = [
            row
            for row in sorted(rows, key=lambda item: item["date"])
            if _as_float(row.get("close")) > 0
        ]
        prices = np.asarray([_as_float(row["close"]) for row in clean_rows], dtype=float)
        dates = [str(row["date"]) for row in clean_rows]
        if prices.shape[0] < 30:
            raise ArtifactValidationError("At least 30 positive close prices are required")
        return prices, dates

    def _daily_return_estimate(
        self,
        *,
        asset_class: str,
        prices: np.ndarray,
        window_size: int,
    ) -> tuple[float, float, dict[str, Any]]:
        log_prices = np.log(np.clip(prices, 1e-12, None))
        returns = np.diff(log_prices)
        max_window = max(int(returns.shape[0]), 1)
        candidate_windows = [
            min(max(int(window_size), 20), max_window),
            min(63, max_window),
            min(126, max_window),
            min(252, max_window),
            max_window,
        ]
        windows = sorted({window for window in candidate_windows if window >= 1})
        means = []
        vols = []
        weights = []
        for window in windows:
            sample = returns[-window:]
            means.append(float(np.mean(sample)))
            vols.append(float(np.std(sample)))
            weights.append(float(window))

        normalized_weights = np.asarray(weights, dtype=float)
        normalized_weights /= max(float(normalized_weights.sum()), 1e-12)
        blended_mu = float(np.average(np.asarray(means), weights=normalized_weights))
        blended_vol = float(np.average(np.asarray(vols), weights=normalized_weights))
        long_mu = float((log_prices[-1] - log_prices[0]) / max(prices.shape[0] - 1, 1))
        medium_start = max(prices.shape[0] - 253, 0)
        medium_mu = float(
            (log_prices[-1] - log_prices[medium_start])
            / max(prices.shape[0] - medium_start - 1, 1)
        )
        short_start = max(prices.shape[0] - 64, 0)
        short_mu = float(
            (log_prices[-1] - log_prices[short_start])
            / max(prices.shape[0] - short_start - 1, 1)
        )
        shrink = 80.0 / (80.0 + float(max(windows)))
        prior_mu = (0.70 * long_mu) + (0.30 * self._cash_daily_return())
        daily_mu = ((1.0 - shrink) * blended_mu) + (shrink * prior_mu)
        daily_mu = (0.55 * daily_mu) + (0.30 * medium_mu) + (0.15 * short_mu)
        lower, upper = _return_caps(asset_class)
        capped_mu = _soft_cap_return(daily_mu, lower=lower, upper=upper)
        recent_vol = float(np.std(returns[-min(90, returns.shape[0]) :])) if returns.size else 0.0
        daily_volatility = max(blended_vol, recent_vol, 1e-6)
        return capped_mu, daily_volatility, {
            "method": "supabase_multi_window_statistical",
            "method_version": FORECAST_METHOD_VERSION,
            "requested_window": int(window_size),
            "projection_windows": windows,
            "shrinkage_to_prior": shrink,
            "long_run_daily_return": long_mu,
            "medium_daily_return": medium_mu,
            "short_daily_return": short_mu,
            "public_lower_daily_return_cap": lower,
            "public_upper_daily_return_cap": upper,
        }

    def _scenario_path(
        self,
        *,
        latest_price: float,
        latest_date: str,
        forecast_start_date: str,
        daily_mu: float,
        daily_volatility: float,
        horizon_days: int,
        z_score: float,
        prices: np.ndarray,
    ) -> list[dict[str, Any]]:
        horizon = max(int(horizon_days), 1)
        historical = np.asarray(prices, dtype=float).reshape(-1)
        recent_window = min(max(horizon, 30), 90, max(historical.shape[0] - 1, 1))
        recent_prices = historical[-(recent_window + 1) :]
        recent_returns = np.diff(np.log(np.clip(recent_prices, 1e-12, None)))
        if recent_returns.shape[0] == 0:
            residuals = np.zeros(horizon, dtype=float)
        else:
            residuals = recent_returns - float(np.mean(recent_returns))
            if float(np.std(residuals)) > 1e-12 and daily_volatility > 1e-12:
                residuals = residuals * (daily_volatility / float(np.std(residuals)))
            seed = (
                int(abs(float(latest_price)) * 10_000)
                ^ (horizon * 1_000_003)
                ^ int((float(z_score) + 3.0) * 100_000)
                ^ int(date.fromisoformat(latest_date).toordinal())
            )
            rng = np.random.default_rng(seed)
            residuals = rng.choice(residuals, size=horizon, replace=True)

        days = np.arange(0, horizon + 1, dtype=float)
        cumulative_noise = np.concatenate(
            [np.array([0.0], dtype=float), np.cumsum(residuals)]
        )
        bridge = cumulative_noise - ((days / float(horizon)) * cumulative_noise[-1])
        noise_scale = 0.32 if abs(z_score) < 1e-12 else 0.42
        path = []
        for index, day in enumerate(days):
            projected = latest_price * math.exp(
                (daily_mu * float(day))
                + (z_score * daily_volatility * math.sqrt(max(float(day), 0.0)))
                + (noise_scale * bridge[index])
            )
            path.append(
                {
                    "day": int(day),
                    "date": _date_add_days(forecast_start_date, int(day)),
                    "price": float(projected),
                }
            )
        return path

    def _payload_from_snapshot(
        self,
        *,
        metadata: dict[str, Any],
        rows: list[dict[str, Any]],
        snapshot: dict[str, Any],
        forecast_start_date: str | None = None,
    ) -> dict[str, Any]:
        ticker = metadata["ticker"]
        asset_class = metadata["asset_class"]
        prices, dates = self._prepare_price_series(rows)
        history_count = min(252, prices.shape[0])
        historical_prices = [
            {"date": date_value, "price": float(price_value)}
            for date_value, price_value in zip(dates[-history_count:], prices[-history_count:])
        ]
        forecast_paths = snapshot.get("forecast_paths_json") or {}
        if isinstance(forecast_paths, str):
            forecast_paths = json.loads(forecast_paths)
        forecast_start_date = forecast_start_date or _forecast_start_date()
        forecast_paths = _rebase_forecast_paths(forecast_paths, forecast_start_date)
        returns = {
            "bear": _as_float(snapshot.get("bear_return")),
            "base": _as_float(snapshot.get("base_return")),
            "bull": _as_float(snapshot.get("bull_return")),
        }
        max_drawdown = _as_float(snapshot.get("drawdown"))
        annualized_volatility = _as_float(snapshot.get("volatility"))
        confidence = _as_float(snapshot.get("confidence"), 0.05)
        confidence_label = snapshot.get("confidence_label") or _confidence_label(confidence)
        target_prices = {
            "bear": _as_float(snapshot.get("bear_target")),
            "base": _as_float(snapshot.get("base_target")),
            "bull": _as_float(snapshot.get("bull_target")),
        }
        risk_label = _risk_label(annualized_volatility, max_drawdown)
        summary = (
            f"{ticker} has a {confidence_label.lower()}-confidence base-case scenario "
            f"with {risk_label.lower()} risk."
        )
        return {
            "ticker": ticker,
            "asset_class": asset_class,
            "latest_date": dates[-1],
            "forecast_start_date": forecast_start_date,
            "latest_price": _as_float(snapshot.get("latest_price"), float(prices[-1])),
            "horizon_days": int(snapshot["horizon_days"]),
            "historical_prices": historical_prices,
            "forecast_paths": forecast_paths,
            "target_prices": target_prices,
            "returns": returns,
            "risk_metrics": {
                "model_estimated_daily_return": _as_float(snapshot.get("base_return"))
                / max(int(snapshot["horizon_days"]), 1),
                "annualized_return": _as_float(snapshot.get("base_return")),
                "annualized_volatility": annualized_volatility,
                "max_historical_drawdown": max_drawdown,
                "forecast_spread": max(returns["bull"] - returns["bear"], 0.0),
                "regime_stability": 1.0,
            },
            "confidence": confidence,
            "confidence_label": confidence_label,
            "risk_label": risk_label,
            "opportunity_score": self._forecast_score(
                {
                    "returns": returns,
                    "risk_metrics": {"annualized_volatility": annualized_volatility},
                    "confidence": confidence,
                },
                risk=0.5,
            ),
            "return_estimator": {
                "method": "stored_supabase_snapshot",
                "method_version": snapshot.get("method_version", FORECAST_METHOD_VERSION),
                "snapshot_as_of": snapshot.get("as_of_date"),
            },
            "literacy": self._literacy_payload(),
            "plain_language": summary,
            "data_as_of": dates[-1],
            "source": "supabase_forecast_snapshot",
            "snapshot_used": True,
        }

    def _market_payload_from_snapshot(
        self,
        *,
        metadata: dict[str, Any],
        snapshot: dict[str, Any],
        forecast_start_date: str,
    ) -> dict[str, Any]:
        ticker = metadata["ticker"]
        returns = {
            "bear": _as_float(snapshot.get("bear_return")),
            "base": _as_float(snapshot.get("base_return")),
            "bull": _as_float(snapshot.get("bull_return")),
        }
        max_drawdown = _as_float(snapshot.get("drawdown"))
        annualized_volatility = _as_float(snapshot.get("volatility"))
        confidence = _as_float(snapshot.get("confidence"), 0.05)
        confidence_label = snapshot.get("confidence_label") or _confidence_label(confidence)
        target_prices = {
            "bear": _as_float(snapshot.get("bear_target")),
            "base": _as_float(snapshot.get("base_target")),
            "bull": _as_float(snapshot.get("bull_target")),
        }
        return {
            "ticker": ticker,
            "asset_class": metadata["asset_class"],
            "latest_date": str(snapshot.get("as_of_date") or ""),
            "forecast_start_date": forecast_start_date,
            "latest_price": _as_float(snapshot.get("latest_price")),
            "horizon_days": int(snapshot["horizon_days"]),
            "target_prices": target_prices,
            "returns": returns,
            "risk_metrics": {
                "model_estimated_daily_return": _as_float(snapshot.get("base_return"))
                / max(int(snapshot["horizon_days"]), 1),
                "annualized_return": _as_float(snapshot.get("base_return")),
                "annualized_volatility": annualized_volatility,
                "max_historical_drawdown": max_drawdown,
                "forecast_spread": max(returns["bull"] - returns["bear"], 0.0),
                "regime_stability": 1.0,
            },
            "confidence": confidence,
            "confidence_label": confidence_label,
            "risk_label": _risk_label(annualized_volatility, max_drawdown),
            "return_estimator": {
                "method": "stored_supabase_snapshot",
                "method_version": snapshot.get("method_version", FORECAST_METHOD_VERSION),
                "snapshot_as_of": snapshot.get("as_of_date"),
            },
            "data_as_of": str(snapshot.get("as_of_date") or ""),
            "source": "supabase_forecast_snapshot",
            "snapshot_used": True,
        }

    def _literacy_payload(self) -> dict[str, str]:
        return {
            "bear_base_bull": "Bear, base, and bull are scenario ranges, not guaranteed prices.",
            "volatility": "Volatility estimates how much the asset may swing.",
            "drawdown": "Drawdown is the largest historical fall from a prior high.",
            "confidence": "Confidence is lower when data is noisy or scenarios are wide.",
        }

    def build_ticker_forecast(
        self,
        *,
        ticker: str,
        horizon_days: int,
        window_size: int = 60,
        prefer_snapshot: bool = True,
        forecast_start_date: str | None = None,
    ) -> dict[str, Any]:
        metadata, rows = self._ticker_rows(ticker)
        prices, dates = self._prepare_price_series(rows)
        forecast_start_date = forecast_start_date or _forecast_start_date()
        if prefer_snapshot:
            snapshot = self.repository.get_forecast_snapshot(
                ticker=metadata["ticker"],
                horizon_days=horizon_days,
                window_size=window_size,
            )
            if snapshot is not None and str(snapshot.get("as_of_date")) == dates[-1]:
                return self._payload_from_snapshot(
                    metadata=metadata,
                    rows=rows,
                    snapshot=snapshot,
                    forecast_start_date=forecast_start_date,
                )

        asset_class = metadata["asset_class"]
        horizon = max(int(horizon_days), 1)
        daily_mu, daily_volatility, estimator = self._daily_return_estimate(
            asset_class=asset_class,
            prices=prices,
            window_size=window_size,
        )
        latest_price = float(prices[-1])
        latest_date = dates[-1]
        scenario_z = 0.84
        bear_path = self._scenario_path(
            latest_price=latest_price,
            latest_date=latest_date,
            forecast_start_date=forecast_start_date,
            daily_mu=daily_mu,
            daily_volatility=daily_volatility,
            horizon_days=horizon,
            z_score=-scenario_z,
            prices=prices,
        )
        base_path = self._scenario_path(
            latest_price=latest_price,
            latest_date=latest_date,
            forecast_start_date=forecast_start_date,
            daily_mu=daily_mu,
            daily_volatility=daily_volatility,
            horizon_days=horizon,
            z_score=0.0,
            prices=prices,
        )
        bull_path = self._scenario_path(
            latest_price=latest_price,
            latest_date=latest_date,
            forecast_start_date=forecast_start_date,
            daily_mu=daily_mu,
            daily_volatility=daily_volatility,
            horizon_days=horizon,
            z_score=scenario_z,
            prices=prices,
        )
        target_prices = {
            "bear": float(bear_path[-1]["price"]),
            "base": float(base_path[-1]["price"]),
            "bull": float(bull_path[-1]["price"]),
        }
        returns = {
            name: float((target / max(latest_price, 1e-12)) - 1.0)
            for name, target in target_prices.items()
        }
        annualized_volatility = float(daily_volatility * math.sqrt(252))
        annualized_return = float(math.exp(daily_mu * 252) - 1.0)
        max_drawdown = _historical_drawdown(prices)
        spread = max(returns["bull"] - returns["bear"], 0.0)
        data_factor = min(float(prices.shape[0]) / 756.0, 1.0)
        volatility_penalty = min(annualized_volatility / 1.10, 1.0)
        spread_penalty = min(spread / 1.80, 1.0)
        horizon_penalty = min(float(horizon) / 365.0, 1.0) * 0.14
        confidence = float(
            np.clip(
                0.15
                + (0.35 * data_factor)
                + (0.25 * (1.0 - volatility_penalty))
                + (0.15 * (1.0 - spread_penalty)),
                0.05,
                0.95,
            )
        )
        confidence = float(np.clip(confidence - horizon_penalty, 0.05, 0.95))
        confidence_label = _confidence_label(confidence)
        risk_label = _risk_label(annualized_volatility, max_drawdown)
        if returns["base"] > 0 and confidence >= 0.45:
            summary = (
                f"{metadata['ticker']} has a positive base-case scenario with "
                f"{risk_label.lower()} risk and {confidence_label.lower()} confidence."
            )
        elif returns["base"] > 0:
            summary = f"{metadata['ticker']} has upside in the base case, but the scenario band is wide."
        else:
            summary = f"{metadata['ticker']} has a weak base-case forecast; treat upside as speculative."
        history_count = min(252, prices.shape[0])
        historical_prices = [
            {"date": date_value, "price": float(price_value)}
            for date_value, price_value in zip(dates[-history_count:], prices[-history_count:])
        ]
        forecast_paths = {"bear": bear_path, "base": base_path, "bull": bull_path}
        return {
            "ticker": metadata["ticker"],
            "asset_class": asset_class,
            "latest_date": latest_date,
            "forecast_start_date": forecast_start_date,
            "latest_price": latest_price,
            "horizon_days": horizon,
            "historical_prices": historical_prices,
            "forecast_paths": forecast_paths,
            "target_prices": target_prices,
            "returns": returns,
            "risk_metrics": {
                "model_estimated_daily_return": daily_mu,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "max_historical_drawdown": max_drawdown,
                "forecast_spread": spread,
                "regime_stability": 1.0,
            },
            "confidence": confidence,
            "confidence_label": confidence_label,
            "risk_label": risk_label,
            "opportunity_score": self._forecast_score(
                {
                    "returns": returns,
                    "risk_metrics": {"annualized_volatility": annualized_volatility},
                    "confidence": confidence,
                },
                risk=0.5,
            ),
            "return_estimator": estimator,
            "literacy": self._literacy_payload(),
            "plain_language": summary,
            "data_as_of": latest_date,
            "source": "supabase_ohlcv",
            "snapshot_used": False,
        }

    def forecast_snapshot_row(self, forecast: dict[str, Any]) -> dict[str, Any]:
        return {
            "ticker": forecast["ticker"],
            "as_of_date": forecast["latest_date"],
            "horizon_days": int(forecast["horizon_days"]),
            "window_size": int(forecast["return_estimator"].get("requested_window", 60)),
            "method_version": FORECAST_METHOD_VERSION,
            "latest_price": forecast["latest_price"],
            "bear_target": forecast["target_prices"]["bear"],
            "base_target": forecast["target_prices"]["base"],
            "bull_target": forecast["target_prices"]["bull"],
            "bear_return": forecast["returns"]["bear"],
            "base_return": forecast["returns"]["base"],
            "bull_return": forecast["returns"]["bull"],
            "volatility": forecast["risk_metrics"]["annualized_volatility"],
            "drawdown": forecast["risk_metrics"]["max_historical_drawdown"],
            "confidence": forecast["confidence"],
            "confidence_label": forecast["confidence_label"],
            "forecast_paths_json": forecast["forecast_paths"],
        }

    def universe_payload(self) -> dict[str, Any]:
        if self._universe_payload_cache is not None:
            return self._universe_payload_cache
        universe = self.repository.list_universe()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        all_tickers = []
        for row in universe:
            entry = {
                "ticker": row["ticker"],
                "asset_class": row["asset_class"],
                "display_name": row.get("display_name"),
                "latest_date": None,
                "latest_price": None,
                "history_days": 0,
                "first_date": None,
                "row_count": 0,
                "has_profile": False,
                "profile_as_of_date": None,
                "min_history_days": row.get("min_history_days"),
            }
            grouped[row["asset_class"]].append(entry)
            all_tickers.append(entry)
        asset_classes = [
            {
                "asset_class": asset_class,
                "tickers": tickers,
                "history_days": 0,
            }
            for asset_class, tickers in sorted(grouped.items())
        ]
        payload = {
            "latest_date": "",
            "supported_asset_classes": [group["asset_class"] for group in asset_classes],
            "asset_classes": asset_classes,
            "tickers": all_tickers,
            "source": "supabase",
            "disclaimer": (
                "Forecasts are educational model estimates, not financial advice "
                "or guaranteed returns."
            ),
        }
        self._universe_payload_cache = payload
        return payload

    def market_indices_payload(self) -> dict[str, Any]:
        rows = self.repository.list_latest_market_indices()
        latest_dates = [str(row.get("as_of_date")) for row in rows if row.get("as_of_date")]
        return {
            "source": "supabase",
            "as_of_date": max(latest_dates) if latest_dates else None,
            "indices": [
                {
                    "symbol": row.get("symbol"),
                    "label": row.get("label") or row.get("symbol"),
                    "display_name": row.get("display_name"),
                    "provider_symbol": row.get("provider_symbol"),
                    "value": _as_float(row.get("value")),
                    "previous_close": row.get("previous_close"),
                    "change": row.get("change"),
                    "change_percent": row.get("change_percent"),
                    "day_open": row.get("day_open"),
                    "day_high": row.get("day_high"),
                    "day_low": row.get("day_low"),
                    "volume": row.get("volume"),
                    "currency": row.get("currency") or "USD",
                    "as_of_date": row.get("as_of_date"),
                    "provider": row.get("provider"),
                    "display_order": row.get("display_order") or 0,
                }
                for row in rows
            ],
            "disclaimer": "Index levels are delayed provider data for market context.",
        }

    def ticker_profile_payload(self, ticker: str) -> dict[str, Any]:
        metadata = self.repository.ticker_metadata(ticker)
        if metadata is None:
            raise ArtifactValidationError(f"Unsupported ticker: {ticker}")
        profile = self.repository.get_latest_profile(metadata["ticker"]) or {}
        coverage = self.repository.coverage_for_ticker(metadata["ticker"])
        fields = {
            "bid": profile.get("bid"),
            "ask": profile.get("ask"),
            "last_sale": profile.get("last_sale") or coverage.get("latest_price"),
            "open": profile.get("day_open"),
            "high": profile.get("day_high"),
            "low": profile.get("day_low"),
            "exchange": profile.get("exchange") or metadata.get("exchange"),
            "sector": metadata.get("sector"),
            "industry": metadata.get("industry"),
            "country": metadata.get("country"),
            "benchmark_group": metadata.get("benchmark_group"),
            "provider_symbol": metadata.get("provider_symbol"),
            "market_cap": profile.get("market_cap"),
            "pe_ratio": profile.get("pe_ratio"),
            "fifty_two_week_high": profile.get("fifty_two_week_high"),
            "fifty_two_week_low": profile.get("fifty_two_week_low"),
            "volume": profile.get("volume"),
            "average_volume": profile.get("average_volume"),
            "margin_requirement": profile.get("margin_requirement"),
            "dividend_frequency": profile.get("dividend_frequency"),
            "dividend_yield": profile.get("dividend_yield"),
            "ex_dividend_date": profile.get("ex_dividend_date"),
        }
        return {
            "ticker": metadata["ticker"],
            "asset_class": metadata["asset_class"],
            "display_name": metadata.get("display_name"),
            "as_of_date": profile.get("as_of_date"),
            "data_as_of": coverage.get("latest_date"),
            "source": "supabase",
            "fields": fields,
        }

    def run_ticker_forecast(
        self,
        *,
        ticker: str,
        horizon_days: int,
        window_size: int = 60,
    ) -> dict[str, Any]:
        forecast_start_date = _forecast_start_date()
        return self.build_ticker_forecast(
            ticker=ticker,
            horizon_days=horizon_days,
            window_size=window_size,
            prefer_snapshot=True,
            forecast_start_date=forecast_start_date,
        )

    def _macro_payload(self) -> dict[str, Any]:
        if self._macro_payload_cache is not None:
            return self._macro_payload_cache
        row = self.repository.get_latest_macro_snapshot()
        if row is None:
            return {"date": "", "global_regime": 1, "macro": []}
        fields = [
            ("VIX Market Volatility", "vix"),
            ("Federal Funds Rate", "federal_funds_rate"),
            ("10-Year Treasury Yield", "treasury_10y"),
            ("Unemployment Rate", "unemployment_rate"),
            ("CPI All Items", "cpi_all_items"),
            ("Recession Indicator", "recession_indicator"),
        ]
        payload = {
            "date": row.get("date", ""),
            "global_regime": 1,
            "macro": [
                {"name": label, "value": _as_float(row.get(key))}
                for label, key in fields
                if row.get(key) is not None
            ],
        }
        self._macro_payload_cache = payload
        return payload

    def run_market_forecast(
        self,
        *,
        horizon_days: int,
        risk: float,
        top_n: int = 10,
        window_size: int = 60,
    ) -> dict[str, Any]:
        forecast_start_date = _forecast_start_date()
        cache_key = (max(int(horizon_days), 1), max(int(window_size), 2), forecast_start_date)
        forecasts = [forecast.copy() for forecast in self._market_forecast_cache.get(cache_key, [])]
        if not forecasts:
            universe = self.repository.list_universe()
            active_metadata = {row["ticker"]: row for row in universe}
            snapshots = self.repository.list_latest_forecast_snapshots(
                horizon_days=horizon_days,
                window_size=window_size,
            )
            if snapshots:
                latest_dates = self.repository.latest_ohlcv_dates_by_ticker()
                for snapshot in snapshots:
                    metadata = active_metadata.get(snapshot.get("ticker"))
                    if metadata is None:
                        continue
                    latest_date = latest_dates.get(str(snapshot.get("ticker")))
                    if latest_date and str(snapshot.get("as_of_date")) != latest_date:
                        continue
                    forecasts.append(
                        self._market_payload_from_snapshot(
                            metadata=metadata,
                            snapshot=snapshot,
                            forecast_start_date=forecast_start_date,
                        )
                    )
            if not forecasts:
                for row in universe:
                    try:
                        forecasts.append(
                            self.build_ticker_forecast(
                                ticker=row["ticker"],
                                horizon_days=horizon_days,
                                window_size=window_size,
                                prefer_snapshot=False,
                                forecast_start_date=forecast_start_date,
                            )
                        )
                    except ArtifactValidationError:
                        continue
            self._market_forecast_cache[cache_key] = [forecast.copy() for forecast in forecasts]
        if not forecasts:
            raise ArtifactValidationError("No Supabase tickers have enough market history")
        for forecast in forecasts:
            forecast["opportunity_score"] = self._forecast_score(forecast, risk=risk)
            forecast["risk_score"] = float(
                forecast["risk_metrics"]["annualized_volatility"]
                + max(-forecast["returns"]["bear"], 0.0)
            )
            forecast["risk_adjusted_score"] = float(
                forecast["opportunity_score"] / max(forecast["risk_score"], 0.05)
            )
        ranked = sorted(forecasts, key=lambda entry: entry["opportunity_score"], reverse=True)
        stable = sorted(
            forecasts,
            key=lambda entry: (
                entry["risk_metrics"]["annualized_volatility"],
                -entry["returns"]["base"],
            ),
        )
        downside = sorted(forecasts, key=lambda entry: entry["returns"]["bear"])
        risk_adjusted = sorted(
            forecasts,
            key=lambda entry: entry["risk_adjusted_score"],
            reverse=True,
        )
        best_base = max(forecasts, key=lambda entry: entry["returns"]["base"])
        return {
            "horizon_days": max(int(horizon_days), 1),
            "risk": float(np.clip(risk, 0.0, 1.0)),
            "ranked_tickers": ranked[: max(int(top_n), 1)],
            "highlights": {
                "best_base_case": best_base,
                "best_risk_adjusted": risk_adjusted[0],
                "highest_downside_risk": downside[0],
                "most_stable": stable[0],
            },
            "macro_snapshot": self._macro_payload(),
            "source": "supabase",
            "disclaimer": (
                "Rankings are scenario estimates based on historical market data "
                "and should not be treated as trading instructions."
            ),
        }

    def run_portfolio_simulation(
        self,
        *,
        amount: float,
        risk: float,
        horizon_days: int,
        selected_tickers: list[str] | None = None,
        window_size: int = 60,
    ) -> dict[str, Any]:
        risk_value = float(np.clip(risk, 0.0, 1.0))
        if selected_tickers:
            forecast_start_date = _forecast_start_date()
            forecasts = [
                self.build_ticker_forecast(
                    ticker=ticker,
                    horizon_days=horizon_days,
                    window_size=window_size,
                    prefer_snapshot=True,
                    forecast_start_date=forecast_start_date,
                )
                for ticker in selected_tickers
            ]
            for forecast in forecasts:
                forecast["opportunity_score"] = self._forecast_score(forecast, risk=risk_value)
            ranked = sorted(forecasts, key=lambda entry: entry["opportunity_score"], reverse=True)
        else:
            ranked = self.run_market_forecast(
                horizon_days=horizon_days,
                risk=risk_value,
                top_n=10,
                window_size=window_size,
            )["ranked_tickers"]
        chosen = ranked[: min(len(ranked), 10)]
        if not chosen:
            raise ArtifactValidationError("No tickers available for portfolio simulation")

        cash_return = float((1.0 + self._cash_daily_return()) ** max(int(horizon_days), 1) - 1.0)
        raw_scores = np.asarray(
            [
                max(
                    forecast["opportunity_score"]
                    + (0.25 * max(forecast["returns"]["base"], 0.0))
                    + 0.10,
                    0.01,
                )
                for forecast in chosen
            ],
            dtype=float,
        )
        raw_scores = np.exp(np.clip(raw_scores * 5.0, -5.0, 5.0))
        cash_weight = float(np.clip(0.28 - (0.23 * risk_value), 0.02, 0.35))
        if float(np.mean([forecast["returns"]["base"] for forecast in chosen])) < cash_return:
            cash_weight = min(0.45, cash_weight + 0.12)
        risky_budget = max(1.0 - cash_weight, 0.0)
        max_asset_weight = 0.18 + (0.07 * risk_value)
        risky_weights = _normalize_weights_with_caps(
            raw_scores,
            max_asset_weight / max(risky_budget, 1e-12),
        )
        risky_weights = risky_weights * risky_budget

        allocations = []
        class_weights: dict[str, float] = {}
        bear_return = cash_weight * cash_return
        base_return = cash_weight * cash_return
        bull_return = cash_weight * cash_return
        confidence_values = []
        for weight, forecast in zip(risky_weights, chosen):
            weight_value = float(weight)
            allocations.append(
                {
                    "ticker": forecast["ticker"],
                    "asset_class": forecast["asset_class"],
                    "weight": weight_value,
                    "amount": float(amount * weight_value),
                    "base_return": float(forecast["returns"]["base"]),
                    "bear_return": float(forecast["returns"]["bear"]),
                    "bull_return": float(forecast["returns"]["bull"]),
                    "confidence": float(forecast["confidence"]),
                }
            )
            class_weights[forecast["asset_class"]] = (
                class_weights.get(forecast["asset_class"], 0.0) + weight_value
            )
            bear_return += weight_value * float(forecast["returns"]["bear"])
            base_return += weight_value * float(forecast["returns"]["base"])
            bull_return += weight_value * float(forecast["returns"]["bull"])
            confidence_values.append(float(forecast["confidence"]))

        if cash_weight > 1e-9:
            allocations.append(
                {
                    "ticker": CASH_TICKER,
                    "asset_class": CASH_ASSET_CLASS,
                    "weight": cash_weight,
                    "amount": float(amount * cash_weight),
                    "base_return": cash_return,
                    "bear_return": cash_return,
                    "bull_return": cash_return,
                    "confidence": 0.95,
                }
            )
            class_weights[CASH_ASSET_CLASS] = cash_weight
            confidence_values.append(0.95)

        total_weight = sum(allocation["weight"] for allocation in allocations)
        if total_weight > 1e-12:
            for allocation in allocations:
                allocation["weight"] = float(allocation["weight"] / total_weight)
                allocation["amount"] = float(allocation["weight"] * amount)
        class_allocations = [
            {
                "asset_class": asset_class,
                "weight": float(weight),
                "amount": float(weight * amount),
            }
            for asset_class, weight in sorted(class_weights.items())
        ]
        trade_plan = [
            {
                "ticker": allocation["ticker"],
                "asset_class": allocation["asset_class"],
                "action": "hold_cash" if allocation["ticker"] == CASH_TICKER else "buy",
                "weight": allocation["weight"],
                "amount": allocation["amount"],
            }
            for allocation in allocations
            if allocation["amount"] >= 1.0
        ]
        warnings = [
            "Forecasts are estimates and should be reviewed before making investment decisions."
        ]
        if max(allocation["weight"] for allocation in allocations) > 0.24:
            warnings.append("One position is above 24%, so concentration risk is elevated.")
        if float(np.mean(confidence_values)) < 0.45:
            warnings.append("Average forecast confidence is low because the scenario band is wide.")
        if cash_weight < 0.03 and risk_value < 0.95:
            warnings.append("Cash is very low for a non-maximum risk setting.")
        return {
            "amount": float(amount),
            "risk": risk_value,
            "horizon_days": max(int(horizon_days), 1),
            "method": "supabase_forecast_ranked_scenario_simulator",
            "summary": {
                "bear_value": float(amount * (1.0 + bear_return)),
                "base_value": float(amount * (1.0 + base_return)),
                "bull_value": float(amount * (1.0 + bull_return)),
                "bear_return": float(bear_return),
                "base_return": float(base_return),
                "bull_return": float(bull_return),
                "average_confidence": float(np.mean(confidence_values)),
            },
            "asset_allocations": allocations,
            "class_allocations": class_allocations,
            "trade_plan": trade_plan,
            "source_forecasts": chosen,
            "warnings": warnings,
        }
