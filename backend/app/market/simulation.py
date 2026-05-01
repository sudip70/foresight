from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


SIMULATION_ASSET_CLASSES = ("stock", "etf", "crypto")


def _normalize_with_caps(scores: np.ndarray, caps: np.ndarray | None = None) -> np.ndarray:
    weights = np.clip(np.asarray(scores, dtype=float), 0.0, None)
    if weights.ndim != 1:
        weights = weights.reshape(-1)
    if weights.size == 0:
        return weights
    if float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights, dtype=float)
    weights = weights / float(weights.sum())

    if caps is None:
        return weights

    caps = np.clip(np.asarray(caps, dtype=float).reshape(-1), 0.0, None)
    if caps.shape != weights.shape:
        raise ValueError("caps must match scores shape")
    if float(caps.sum()) < 1.0 - 1e-12:
        caps = caps / max(float(caps.sum()), 1e-12)

    capped = np.minimum(weights, caps)
    for _ in range(len(capped) + 1):
        deficit = 1.0 - float(capped.sum())
        if deficit <= 1e-12:
            return capped / max(float(capped.sum()), 1e-12)
        capacity = caps - capped
        eligible = capacity > 1e-12
        if not bool(np.any(eligible)):
            break
        basis = capped[eligible].copy()
        if float(basis.sum()) <= 1e-12:
            basis = capacity[eligible]
        addition = deficit * (basis / max(float(basis.sum()), 1e-12))
        capped[eligible] += np.minimum(addition, capacity[eligible])

    total = float(capped.sum())
    if total > 1e-12:
        return capped / total
    return np.ones_like(capped, dtype=float) / max(capped.size, 1)


def simulation_class_priors(risk: float) -> dict[str, float]:
    risk_value = float(np.clip(risk, 0.0, 1.0))
    crypto = 0.03 + (0.22 * risk_value)
    etf = 0.45 - (0.20 * risk_value)
    stock = max(1.0 - crypto - etf, 0.0)
    return {
        "stock": float(stock),
        "etf": float(etf),
        "crypto": float(crypto),
    }


def _simulation_class_slots(risk: float, limit: int) -> dict[str, int]:
    risk_value = float(np.clip(risk, 0.0, 1.0))
    if risk_value < 0.35:
        slots = {"stock": 5, "etf": 4, "crypto": 1}
    elif risk_value < 0.75:
        slots = {"stock": 5, "etf": 3, "crypto": 2}
    else:
        slots = {"stock": 5, "etf": 2, "crypto": 3}

    total = sum(slots.values())
    if total == limit:
        return slots
    if total < limit:
        slots["stock"] += limit - total
        return slots

    excess = total - limit
    for asset_class in ("crypto", "etf", "stock"):
        removable = min(excess, max(slots[asset_class] - 1, 0))
        slots[asset_class] -= removable
        excess -= removable
        if excess <= 0:
            break
    return slots


def select_diversified_simulation_forecasts(
    ranked: list[dict[str, Any]],
    *,
    risk: float,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not ranked:
        return []

    limit = max(int(limit), 1)
    index_by_ticker = {str(forecast.get("ticker")): index for index, forecast in enumerate(ranked)}
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for forecast in ranked:
        asset_class = str(forecast.get("asset_class", "")).lower()
        if asset_class in SIMULATION_ASSET_CLASSES:
            by_class[asset_class].append(forecast)

    chosen: list[dict[str, Any]] = []
    chosen_tickers: set[str] = set()
    for asset_class, slot_count in _simulation_class_slots(risk, limit).items():
        for forecast in by_class.get(asset_class, [])[:slot_count]:
            ticker = str(forecast.get("ticker"))
            if ticker in chosen_tickers:
                continue
            chosen.append(forecast)
            chosen_tickers.add(ticker)
            if len(chosen) >= limit:
                break
        if len(chosen) >= limit:
            break

    for forecast in ranked:
        if len(chosen) >= limit:
            break
        ticker = str(forecast.get("ticker"))
        if ticker in chosen_tickers:
            continue
        chosen.append(forecast)
        chosen_tickers.add(ticker)

    return sorted(chosen, key=lambda forecast: index_by_ticker.get(str(forecast.get("ticker")), 0))


def allocate_simulation_risky_weights(
    chosen: list[dict[str, Any]],
    raw_scores: np.ndarray,
    *,
    risky_budget: float,
    risk: float,
    max_asset_weight: float,
) -> np.ndarray:
    raw_scores = np.clip(np.asarray(raw_scores, dtype=float).reshape(-1), 0.0, None)
    if raw_scores.shape[0] != len(chosen):
        raise ValueError("raw_scores must match chosen forecast count")
    if not chosen or risky_budget <= 1e-12:
        return np.zeros(raw_scores.shape[0], dtype=float)

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, forecast in enumerate(chosen):
        asset_class = str(forecast.get("asset_class", "")).lower()
        grouped_indices[asset_class].append(index)

    priors = simulation_class_priors(risk)
    available_classes = [
        asset_class
        for asset_class in SIMULATION_ASSET_CLASSES
        if grouped_indices.get(asset_class)
    ]
    if not available_classes:
        return _normalize_with_caps(
            raw_scores,
            np.full(raw_scores.shape[0], max_asset_weight / max(float(risky_budget), 1e-12)),
        ) * float(risky_budget)

    class_scores = []
    for asset_class in available_classes:
        indices = grouped_indices[asset_class]
        class_score = float(np.mean(raw_scores[indices])) if indices else 0.0
        class_scores.append(priors.get(asset_class, 0.0) * np.sqrt(max(class_score, 1e-12)))
    class_shares = _normalize_with_caps(np.asarray(class_scores, dtype=float))

    weights = np.zeros(raw_scores.shape[0], dtype=float)
    for asset_class, class_share in zip(available_classes, class_shares):
        indices = np.asarray(grouped_indices[asset_class], dtype=int)
        class_budget = float(risky_budget) * float(class_share)
        caps = np.full(
            indices.shape[0],
            float(max_asset_weight) / max(class_budget, 1e-12),
            dtype=float,
        )
        weights[indices] = _normalize_with_caps(raw_scores[indices], caps) * class_budget

    total = float(weights.sum())
    if total > 1e-12:
        weights *= float(risky_budget) / total
    return weights
