from __future__ import annotations

import numpy as np

from backend.app.market.simulation import (
    allocate_simulation_risky_weights,
    select_diversified_simulation_forecasts,
)


def _forecast(ticker: str, asset_class: str, score: float) -> dict:
    return {
        "ticker": ticker,
        "asset_class": asset_class,
        "opportunity_score": score,
        "returns": {"base": 0.10, "bear": -0.05, "bull": 0.20},
        "confidence": 0.65,
    }


def test_default_simulator_selection_keeps_cross_class_candidates():
    ranked = [
        *[_forecast(f"S{index}", "stock", 1.0 - (index * 0.01)) for index in range(12)],
        *[_forecast(f"E{index}", "etf", 0.5 - (index * 0.01)) for index in range(4)],
        *[_forecast(f"C{index}", "crypto", 0.4 - (index * 0.01)) for index in range(3)],
    ]

    chosen = select_diversified_simulation_forecasts(ranked, risk=0.5, limit=10)
    classes = [forecast["asset_class"] for forecast in chosen]

    assert classes.count("stock") == 5
    assert classes.count("etf") == 3
    assert classes.count("crypto") == 2


def test_default_simulator_weights_apply_risk_aware_class_budgets():
    chosen = [
        *[_forecast(f"S{index}", "stock", 1.0) for index in range(5)],
        *[_forecast(f"E{index}", "etf", 0.7) for index in range(3)],
        *[_forecast(f"C{index}", "crypto", 0.6) for index in range(2)],
    ]
    raw_scores = np.ones(len(chosen), dtype=float)

    weights = allocate_simulation_risky_weights(
        chosen,
        raw_scores,
        risky_budget=0.835,
        risk=0.5,
        max_asset_weight=0.215,
    )
    class_weights = {}
    for weight, forecast in zip(weights, chosen):
        class_weights[forecast["asset_class"]] = (
            class_weights.get(forecast["asset_class"], 0.0) + float(weight)
        )

    assert abs(float(weights.sum()) - 0.835) < 1e-9
    assert class_weights["stock"] > 0.35
    assert class_weights["etf"] > 0.20
    assert class_weights["crypto"] > 0.08
