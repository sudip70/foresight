from __future__ import annotations

import math
from typing import Any

import numpy as np


def historical_drawdown(prices: np.ndarray) -> float:
    series = np.asarray(prices, dtype=float).reshape(-1)
    if series.shape[0] <= 1:
        return 0.0
    running_peak = np.maximum.accumulate(series)
    drawdown = 1.0 - (series / np.clip(running_peak, 1e-12, None))
    return float(np.max(drawdown))


def confidence_label(confidence: float) -> str:
    if confidence >= 0.70:
        return "High"
    if confidence >= 0.45:
        return "Medium"
    return "Low"


def risk_label(annualized_volatility: float, max_drawdown: float) -> str:
    risk_score = annualized_volatility + (0.60 * max_drawdown)
    if risk_score >= 0.65:
        return "High"
    if risk_score >= 0.30:
        return "Moderate"
    return "Lower"


def soft_cap_return(value: float, *, lower: float, upper: float) -> float:
    if value >= 0.0:
        return float(upper * math.tanh(value / max(upper, 1e-12)))
    lower_abs = abs(lower)
    return float(-lower_abs * math.tanh(abs(value) / max(lower_abs, 1e-12)))


def return_caps(asset_class: str) -> tuple[float, float]:
    """Return (lower_cap, upper_cap) daily return bounds for public-facing scenario outputs."""
    if asset_class == "crypto":
        return -0.0030, 0.00125
    if asset_class == "stock":
        return -0.0016, 0.00075
    return -0.0012, 0.00055


def forecast_score(forecast: dict[str, Any], *, risk: float) -> float:
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


def redistribute_weight_excess(
    weights: np.ndarray,
    caps: np.ndarray,
    cash_weight: float,
    *,
    eligible: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    adjusted = np.asarray(weights, dtype=float).copy()
    cap_values = np.asarray(caps, dtype=float)
    eligible_mask = (
        np.ones_like(adjusted, dtype=bool)
        if eligible is None
        else np.asarray(eligible, dtype=bool)
    )
    for _ in range(20):
        over = adjusted > cap_values
        if not bool(np.any(over)):
            break
        excess = float(np.sum(adjusted[over] - cap_values[over]))
        adjusted[over] = cap_values[over]
        capacity = np.clip(cap_values - adjusted, 0.0, None)
        capacity[~eligible_mask] = 0.0
        capacity_sum = float(np.sum(capacity))
        if excess <= 1e-12:
            break
        if capacity_sum <= 1e-12:
            cash_weight += excess
            break
        adjusted += excess * (capacity / capacity_sum)
    return adjusted, cash_weight


def apply_portfolio_weight_constraints(
    *,
    chosen: list[dict[str, Any]],
    risky_weights: np.ndarray,
    cash_weight: float,
    max_single_position_weight: float,
    max_crypto_weight: float | None,
    constraints: dict[str, Any],
) -> tuple[np.ndarray, float]:
    weights = np.asarray(risky_weights, dtype=float)
    caps = np.full_like(weights, max_single_position_weight, dtype=float)
    if (
        constraints.get("max_single_position_weight") is not None
        and bool(np.any(weights > caps + 1e-9))
    ):
        constraints["binding"].append("max_single_position_weight")
    weights, cash_weight = redistribute_weight_excess(weights, caps, cash_weight)
    if max_crypto_weight is not None:
        crypto_mask = np.asarray(
            [forecast.get("asset_class") == "crypto" for forecast in chosen],
            dtype=bool,
        )
        crypto_weight = float(np.sum(weights[crypto_mask]))
        if crypto_weight > max_crypto_weight + 1e-9:
            constraints["binding"].append("max_crypto_weight")
            excess = crypto_weight - max_crypto_weight
            weights[crypto_mask] *= max_crypto_weight / max(crypto_weight, 1e-12)
            non_crypto = ~crypto_mask
            capacity = np.clip(caps - weights, 0.0, None)
            capacity[~non_crypto] = 0.0
            capacity_sum = float(np.sum(capacity))
            if capacity_sum <= 1e-12:
                cash_weight += excess
            else:
                weights += excess * (capacity / capacity_sum)
            weights, cash_weight = redistribute_weight_excess(
                weights,
                caps,
                cash_weight,
                eligible=non_crypto,
            )
    total = float(np.sum(weights) + cash_weight)
    if total > 1.0 + 1e-9:
        scale = max((1.0 - cash_weight) / max(float(np.sum(weights)), 1e-12), 0.0)
        weights *= scale
    elif total < 1.0 - 1e-9:
        cash_weight += 1.0 - total
    return weights, float(cash_weight)


def portfolio_constraint_payload(
    *,
    max_crypto_weight: float | None,
    max_single_position_weight: float | None,
    min_cash_weight: float | None,
    preferred_asset_classes: list[str] | None,
) -> dict[str, Any]:
    preferred = {
        str(asset_class).strip().lower()
        for asset_class in preferred_asset_classes or []
        if str(asset_class).strip().lower() in {"stock", "etf", "crypto"}
    }
    return {
        "max_crypto_weight": None
        if max_crypto_weight is None
        else float(np.clip(max_crypto_weight, 0.0, 1.0)),
        "max_single_position_weight": None
        if max_single_position_weight is None
        else float(np.clip(max_single_position_weight, 0.01, 1.0)),
        "min_cash_weight": None
        if min_cash_weight is None
        else float(np.clip(min_cash_weight, 0.0, 1.0)),
        "preferred_asset_classes": sorted(preferred),
        "binding": [],
    }
