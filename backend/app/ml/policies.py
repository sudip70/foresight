from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from backend.app.ml.envs import SINGLE_AGENT_MARKET_BLOCKS
from backend.app.ml.feature_groups import build_feature_slices
from backend.app.ml.numpy_compat import install_numpy_pickle_compat

try:  # gymnasium is only needed when loading trained SB3 models.
    from gymnasium import spaces
except ImportError:  # pragma: no cover - slim deployment guard
    spaces = None


class PolicyLoadError(RuntimeError):
    """Raised when a policy bundle cannot be loaded."""


@dataclass(frozen=True)
class CashOverlayParams:
    """Scale factors convert daily return/volatility gaps into cash allocation pressure."""

    target_return_risk_neutral_scale: float = 170.0
    target_return_risk_scale_reduction: float = 40.0
    risky_mean_carry_risk_neutral_scale: float = 620.0
    risky_mean_carry_risk_scale_reduction: float = 180.0
    risky_median_carry_risk_neutral_scale: float = 420.0
    risky_median_carry_risk_scale_reduction: float = 120.0
    volatility_risk_neutral_scale: float = 8.0
    volatility_risk_scale_reduction: float = 2.0
    positive_return_relief_base_scale: float = 180.0
    positive_return_relief_risk_scale: float = 140.0


DEFAULT_CASH_OVERLAY_PARAMS = CashOverlayParams()


def normalize_weights(weights: np.ndarray, *, max_weight: float | None = None) -> np.ndarray:
    clipped = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, None)
    total = clipped.sum()
    if total <= 0:
        normalized = np.ones_like(clipped) / len(clipped)
    else:
        normalized = clipped / total

    if max_weight is None:
        return normalized

    cap = max(float(max_weight), 1.0 / len(normalized))
    capped = normalized.copy()
    for _ in range(len(capped) + 1):
        over_cap = capped > cap
        if not over_cap.any():
            return capped / capped.sum()

        excess = float((capped[over_cap] - cap).sum())
        capped[over_cap] = cap
        under_cap = ~over_cap
        capacity = cap - capped[under_cap]
        if not under_cap.any() or capacity.sum() <= 1e-12:
            return capped / capped.sum()

        redistributor = capped[under_cap].copy()
        if redistributor.sum() <= 1e-12:
            redistributor = capacity
        capped[under_cap] += excess * (redistributor / redistributor.sum())

    return capped / capped.sum()


def normalize_weights_with_caps(weights: np.ndarray, max_weights: np.ndarray) -> np.ndarray:
    normalized = normalize_weights(weights)
    caps = np.asarray(max_weights, dtype=float).reshape(-1)
    if caps.shape != normalized.shape:
        raise ValueError("max_weights must match weights shape")
    caps = np.clip(caps, 0.0, None)
    if float(caps.sum()) < 1.0 - 1e-12:
        raise ValueError("Per-asset max weights must sum to at least 1.0")

    capped = np.minimum(normalized, caps)
    for _ in range(len(capped) + 1):
        deficit = 1.0 - float(capped.sum())
        if deficit <= 1e-12:
            return capped / capped.sum()

        capacity = caps - capped
        eligible = capacity > 1e-12
        if not eligible.any():
            break

        basis = capped[eligible].copy()
        if float(basis.sum()) <= 1e-12:
            basis = capacity[eligible]
        addition = deficit * (basis / basis.sum())
        capped[eligible] += np.minimum(addition, capacity[eligible])

    return normalize_weights(capped)


def apply_cash_risk_off_overlay(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    *,
    cash_index: int,
    max_cash_weight: float,
    target_return: float = 0.0,
) -> tuple[np.ndarray, dict | None]:
    adjusted = normalize_weights(weights)
    mu = np.asarray(expected_returns, dtype=float).reshape(-1)
    if mu.shape != adjusted.shape:
        raise ValueError("expected_returns must match weights shape")

    cash_index = int(cash_index)
    if cash_index < 0 or cash_index >= adjusted.shape[0]:
        raise ValueError("cash_index is out of bounds")

    current_expected = float(np.dot(adjusted, mu))
    if current_expected >= target_return - 1e-12:
        return adjusted, None

    risky_mask = np.ones(adjusted.shape[0], dtype=bool)
    risky_mask[cash_index] = False
    risky_weight = float(adjusted[risky_mask].sum())
    cash_weight = float(adjusted[cash_index])
    cash_capacity = max(float(max_cash_weight) - cash_weight, 0.0)
    if risky_weight <= 1e-12 or cash_capacity <= 1e-12:
        return adjusted, None

    risky_expected = float(np.dot(adjusted[risky_mask], mu[risky_mask]))
    risky_mean_return = risky_expected / risky_weight
    cash_return = float(mu[cash_index])
    expected_improvement_per_shift = cash_return - risky_mean_return
    if expected_improvement_per_shift <= 1e-12:
        return adjusted, None

    required_shift = (float(target_return) - current_expected) / expected_improvement_per_shift
    shift = min(max(required_shift, 0.0), cash_capacity, risky_weight)
    if shift <= 1e-12:
        return adjusted, None

    next_weights = adjusted.copy()
    next_weights[risky_mask] *= (risky_weight - shift) / risky_weight
    next_weights[cash_index] += shift
    next_weights = normalize_weights(next_weights)
    next_expected = float(np.dot(next_weights, mu))

    return next_weights, {
        "applied": True,
        "reason": "negative_expected_return",
        "pre_expected_daily_return": current_expected,
        "target_daily_return": float(target_return),
        "post_expected_daily_return": next_expected,
        "cash_weight_before": cash_weight,
        "cash_weight_after": float(next_weights[cash_index]),
        "cash_shift": float(shift),
    }


def apply_cash_risk_managed_overlay(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_diag: np.ndarray,
    *,
    cash_index: int,
    max_cash_weight: float,
    risk_appetite: float,
    cash_prior: float | None = None,
    target_return: float = 0.0,
    params: CashOverlayParams = DEFAULT_CASH_OVERLAY_PARAMS,
) -> tuple[np.ndarray, dict | None]:
    adjusted = normalize_weights(weights)
    mu = np.asarray(expected_returns, dtype=float).reshape(-1)
    variance_diag = np.clip(np.asarray(covariance_diag, dtype=float).reshape(-1), 0.0, None)
    if mu.shape != adjusted.shape or variance_diag.shape != adjusted.shape:
        raise ValueError("expected_returns and covariance_diag must match weights shape")

    cash_index = int(cash_index)
    if cash_index < 0 or cash_index >= adjusted.shape[0]:
        raise ValueError("cash_index is out of bounds")

    risky_mask = np.ones(adjusted.shape[0], dtype=bool)
    risky_mask[cash_index] = False
    risky_weight = float(adjusted[risky_mask].sum())
    cash_weight = float(adjusted[cash_index])
    if risky_weight <= 1e-12:
        return adjusted, None

    risk_value = float(np.clip(risk_appetite, 0.0, 1.0))
    current_expected = float(np.dot(adjusted, mu))
    positive_share = float(np.mean(mu[risky_mask] > 0.0))
    current_volatility = float(np.sqrt(max(np.dot(adjusted**2, variance_diag), 0.0)))
    cash_return = float(mu[cash_index])
    risky_mean_return = float(np.mean(mu[risky_mask]))
    risky_median_return = float(np.median(mu[risky_mask]))
    effective_target_return = float(target_return) - (0.00045 * risk_value)
    base_cash_target = 0.03 + ((1.0 - risk_value) * 0.20)
    if cash_prior is not None:
        base_cash_target = max(
            base_cash_target,
            float(np.clip(cash_prior, 0.0, 1.0)) * (0.30 + (0.20 * (1.0 - risk_value))),
        )

    expected_return_gap = effective_target_return - current_expected
    expected_return_pressure_scale = (
        params.target_return_risk_neutral_scale
        - (params.target_return_risk_scale_reduction * risk_value)
    )
    expected_return_pressure = np.clip(
        expected_return_gap * expected_return_pressure_scale,
        0.0,
        0.22,
    )
    breadth_gap = 0.55 - positive_share
    breadth_pressure = np.clip(breadth_gap * 0.35, 0.0, 0.12)
    risky_mean_carry_gap = cash_return - risky_mean_return
    risky_mean_carry_pressure_scale = (
        params.risky_mean_carry_risk_neutral_scale
        - (params.risky_mean_carry_risk_scale_reduction * risk_value)
    )
    carry_pressure = np.clip(
        risky_mean_carry_gap * risky_mean_carry_pressure_scale,
        0.0,
        0.32,
    )
    risky_median_carry_gap = cash_return - risky_median_return
    risky_median_carry_pressure_scale = (
        params.risky_median_carry_risk_neutral_scale
        - (params.risky_median_carry_risk_scale_reduction * risk_value)
    )
    median_pressure = np.clip(
        risky_median_carry_gap * risky_median_carry_pressure_scale,
        0.0,
        0.18,
    )
    target_volatility = 0.007 + (0.013 * risk_value)
    volatility_gap = current_volatility - target_volatility
    volatility_pressure_scale = (
        params.volatility_risk_neutral_scale
        - (params.volatility_risk_scale_reduction * risk_value)
    )
    volatility_pressure = np.clip(
        volatility_gap * volatility_pressure_scale,
        0.0,
        0.12,
    )
    positive_return_relief_scale = (
        params.positive_return_relief_base_scale
        + (params.positive_return_relief_risk_scale * risk_value)
    )
    positive_relief = np.clip(
        max(current_expected - max(effective_target_return, -0.00045), 0.0)
        * positive_return_relief_scale,
        0.0,
        0.20,
    )
    target_cash = np.clip(
        base_cash_target
        + expected_return_pressure
        + breadth_pressure
        + carry_pressure
        + median_pressure
        + volatility_pressure
        - positive_relief,
        0.02,
        float(max_cash_weight),
    )
    if cash_return > risky_mean_return and positive_share < 0.60:
        target_cash = max(
            target_cash,
            min(float(max_cash_weight), 0.55 + ((1.0 - risk_value) * 0.30)),
        )

    if abs(target_cash - cash_weight) <= 1e-3:
        return adjusted, None

    next_weights = adjusted.copy()
    if target_cash > cash_weight:
        shift = min(target_cash - cash_weight, risky_weight)
        next_weights[risky_mask] *= (risky_weight - shift) / risky_weight
        next_weights[cash_index] += shift
    else:
        deploy = min(cash_weight - target_cash, cash_weight)
        basis = next_weights[risky_mask].copy()
        if float(basis.sum()) <= 1e-12:
            basis = np.clip(mu[risky_mask], 0.0, None)
        if float(basis.sum()) <= 1e-12:
            basis = np.ones(int(risky_mask.sum()), dtype=float)
        next_weights[risky_mask] += deploy * (basis / basis.sum())
        next_weights[cash_index] -= deploy

    next_weights = normalize_weights(next_weights)
    next_expected = float(np.dot(next_weights, mu))
    next_volatility = float(np.sqrt(max(np.dot(next_weights**2, variance_diag), 0.0)))
    return next_weights, {
        "applied": True,
        "reason": "risk_managed_cash_target",
        "pre_expected_daily_return": current_expected,
        "post_expected_daily_return": next_expected,
        "pre_volatility": current_volatility,
        "post_volatility": next_volatility,
        "target_daily_return": effective_target_return,
        "positive_asset_share": positive_share,
        "cash_daily_return": cash_return,
        "risky_mean_daily_return": risky_mean_return,
        "risky_median_daily_return": risky_median_return,
        "expected_return_pressure": float(expected_return_pressure),
        "breadth_pressure": float(breadth_pressure),
        "carry_pressure": float(carry_pressure),
        "median_pressure": float(median_pressure),
        "volatility_pressure": float(volatility_pressure),
        "positive_return_relief": float(positive_relief),
        "cash_weight_before": cash_weight,
        "cash_weight_after": float(next_weights[cash_index]),
        "cash_target": float(target_cash),
        "risk_appetite": risk_value,
        "cash_prior": None if cash_prior is None else float(cash_prior),
    }


def normalize_action_with_cash_sleeve(
    weights: np.ndarray,
    *,
    risky_asset_count: int,
    max_risky_weight: float | None = None,
    max_cash_weight: float | None = None,
) -> np.ndarray:
    vector = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, None)
    risky_asset_count = int(risky_asset_count)
    if risky_asset_count <= 0:
        raise ValueError("risky_asset_count must be positive")
    if vector.shape[0] <= risky_asset_count:
        return normalize_weights(vector[:risky_asset_count], max_weight=max_risky_weight)

    risky_weights = vector[:risky_asset_count]
    cash_weight = np.array([float(vector[risky_asset_count:].sum())], dtype=float)
    combined = np.concatenate([risky_weights, cash_weight])
    caps = np.ones_like(combined)
    if max_risky_weight is not None:
        caps[:risky_asset_count] = float(max_risky_weight)
    if max_cash_weight is not None:
        caps[-1] = float(max_cash_weight)
    return normalize_weights_with_caps(combined, caps)


def apply_class_guardrails(
    weights: np.ndarray,
    *,
    class_ranges: dict[str, tuple[int, int]],
    max_class_weights: dict[str, float] | None = None,
    max_asset_weight: float | None = None,
) -> np.ndarray:
    adjusted = normalize_weights(weights, max_weight=max_asset_weight)
    if not max_class_weights:
        return adjusted

    capped_classes: set[str] = set()
    for _ in range(len(max_class_weights) + 1):
        excess = 0.0
        for asset_class, max_weight in max_class_weights.items():
            if asset_class not in class_ranges:
                continue
            start, end = class_ranges[asset_class]
            class_weight = float(adjusted[start:end].sum())
            cap = float(max_weight)
            if class_weight > cap:
                if class_weight > 1e-12:
                    adjusted[start:end] *= cap / class_weight
                excess += class_weight - cap
                capped_classes.add(asset_class)

        if excess <= 1e-12:
            return normalize_weights(adjusted, max_weight=max_asset_weight)

        eligible_indices: list[int] = []
        capacities: list[float] = []
        for asset_class, (start, end) in class_ranges.items():
            if asset_class in capped_classes:
                continue
            class_cap = max_class_weights.get(asset_class, 1.0)
            class_weight = float(adjusted[start:end].sum())
            class_capacity = max(float(class_cap) - class_weight, 0.0)
            if class_capacity <= 1e-12:
                continue
            width = end - start
            eligible_indices.extend(range(start, end))
            capacities.extend([class_capacity / max(width, 1)] * width)

        if not eligible_indices or sum(capacities) <= 1e-12:
            return normalize_weights(adjusted, max_weight=max_asset_weight)

        indices = np.asarray(eligible_indices, dtype=int)
        capacity = np.asarray(capacities, dtype=float)
        current = adjusted[indices]
        basis = current if current.sum() > 1e-12 else capacity
        allocation = excess * (basis / basis.sum())
        adjusted[indices] += np.minimum(allocation, capacity)
        adjusted = normalize_weights(adjusted, max_weight=max_asset_weight)

    return normalize_weights(adjusted, max_weight=max_asset_weight)


def policy_signal_blend_weight(
    metadata: dict,
    *,
    alpha_key: str = "eval_mean_benchmark_alpha",
    sharpe_key: str = "eval_mean_sharpe",
    final_value_key: str = "eval_mean_final_value",
    base_blend: float = 0.10,
    max_blend: float = 0.85,
) -> float:
    alpha = float(metadata.get(alpha_key, 0.0) or 0.0)
    sharpe = float(metadata.get(sharpe_key, 0.0) or 0.0)
    final_value = float(metadata.get(final_value_key, 1.0) or 1.0)

    blend = float(base_blend)
    if alpha < 0.0:
        blend += min(0.35, abs(alpha) * 1_250.0)
    if sharpe < 0.75:
        blend += min(0.25, (0.75 - sharpe) * 0.35)
    if final_value < 1.0:
        blend += min(0.15, (1.0 - final_value) * 1.5)
    return float(np.clip(blend, 0.0, max_blend))


def blend_allocation_sources(
    primary_weights: np.ndarray,
    secondary_weights: np.ndarray,
    *,
    secondary_weight: float,
) -> np.ndarray:
    primary = normalize_weights(primary_weights)
    secondary = normalize_weights(secondary_weights)
    if primary.shape != secondary.shape:
        raise ValueError("Allocation sources must have the same shape")

    blend = float(np.clip(secondary_weight, 0.0, 1.0))
    return normalize_weights(((1.0 - blend) * primary) + (blend * secondary))


def constrain_turnover(
    target_weights: np.ndarray,
    previous_weights: np.ndarray,
    *,
    max_turnover: float | None,
) -> np.ndarray:
    target = normalize_weights(target_weights)
    previous = normalize_weights(previous_weights)
    if max_turnover is None:
        return target

    turnover_cap = max(float(max_turnover), 0.0)
    delta = target - previous
    turnover = float(np.sum(np.abs(delta)))
    if turnover <= turnover_cap or turnover <= 1e-12:
        return target

    scaled = previous + (delta * (turnover_cap / turnover))
    return normalize_weights(scaled)


def align_observation(observation: np.ndarray, target_dim: int | None) -> np.ndarray:
    vector = np.asarray(observation, dtype=np.float32).reshape(-1)
    if target_dim is None or target_dim == vector.shape[0]:
        return vector
    if vector.shape[0] < target_dim:
        return np.concatenate(
            [vector, np.zeros(target_dim - vector.shape[0], dtype=np.float32)]
        )
    return vector[:target_dim]


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=float).reshape(-1), -8.0, 8.0)
    shifted = clipped - float(np.max(clipped))
    return normalize_weights(np.exp(shifted))


def _single_agent_market_blocks(
    observation: np.ndarray,
    *,
    n_assets: int,
) -> list[np.ndarray]:
    cursor = 0
    blocks: list[np.ndarray] = []
    for _ in range(int(SINGLE_AGENT_MARKET_BLOCKS)):
        blocks.append(np.asarray(observation[cursor : cursor + n_assets], dtype=float))
        cursor += n_assets
    return blocks


@dataclass
class FixedWeightPolicy:
    weights: np.ndarray
    observation_dim: int | None = None

    def predict(self, observation: np.ndarray) -> np.ndarray:
        align_observation(observation, self.observation_dim)
        return normalize_weights(self.weights)


@dataclass
class LinearPolicy:
    weights_matrix: np.ndarray
    bias: np.ndarray
    observation_dim: int

    def predict(self, observation: np.ndarray) -> np.ndarray:
        aligned = align_observation(observation, self.observation_dim)
        raw = self.weights_matrix @ aligned + self.bias
        return normalize_weights(raw)


@dataclass
class SingleAgentSignalPolicy:
    action_dim: int
    observation_dim: int
    cash_enabled: bool = True

    def predict(self, observation: np.ndarray) -> np.ndarray:
        aligned = align_observation(observation, self.observation_dim)
        n_assets = int(self.action_dim)
        risky_asset_count = n_assets - 1 if self.cash_enabled and n_assets > 1 else n_assets
        blocks = _single_agent_market_blocks(aligned, n_assets=n_assets)
        (
            return_1d,
            return_5d,
            return_21d,
            return_63d,
            vol_21d,
            vol_63d,
            intraday_range,
            close_to_open,
            volume_ratio,
            drawdown_63d,
            sma_ratio_21d,
            previous_weights,
        ) = blocks

        cursor = int(SINGLE_AGENT_MARKET_BLOCKS) * n_assets
        risk_appetite = float(np.clip(aligned[cursor + 1], 0.0, 1.0))
        regime = np.asarray(aligned[cursor + 2 : cursor + 5], dtype=float)
        bull = float(regime[0]) if regime.size >= 1 else 0.0
        bear = float(regime[2]) if regime.size >= 3 else 0.0

        previous_risky = previous_weights[:risky_asset_count]
        drawdown_pressure = np.clip(-drawdown_63d[:risky_asset_count], 0.0, None)
        trend = (
            (0.25 * return_1d[:risky_asset_count])
            + (0.40 * return_5d[:risky_asset_count])
            + (0.85 * return_21d[:risky_asset_count])
            + (0.95 * return_63d[:risky_asset_count])
            + (0.65 * sma_ratio_21d[:risky_asset_count])
            + (0.12 * close_to_open[:risky_asset_count])
            + (0.10 * volume_ratio[:risky_asset_count])
            + (0.20 * previous_risky)
        )
        risk_penalty = (
            (0.85 * vol_21d[:risky_asset_count])
            + (0.55 * vol_63d[:risky_asset_count])
            + (0.45 * intraday_range[:risky_asset_count])
            + (0.90 * drawdown_pressure)
        )
        regime_tilt = (
            bull
            * (
                (0.25 * return_21d[:risky_asset_count])
                + (0.15 * return_63d[:risky_asset_count])
                + (0.12 * sma_ratio_21d[:risky_asset_count])
            )
        ) - (
            bear
            * (
                (0.55 * drawdown_pressure)
                + (0.35 * vol_21d[:risky_asset_count])
            )
        )
        risk_tilt = (
            risk_appetite
            * (
                (0.10 * return_1d[:risky_asset_count])
                + (0.18 * return_5d[:risky_asset_count])
                + (0.15 * sma_ratio_21d[:risky_asset_count])
            )
        ) - ((1.0 - risk_appetite) * ((0.15 * vol_21d[:risky_asset_count]) + (0.10 * drawdown_pressure)))
        scores = np.clip(trend + regime_tilt + risk_tilt - risk_penalty, -5.0, 5.0)
        temperature = 1.8 + (1.8 * risk_appetite)
        asset_logits = scores * temperature

        if not self.cash_enabled or risky_asset_count == n_assets:
            return _softmax_logits(asset_logits)

        previous_cash = float(previous_weights[risky_asset_count])
        signal_strength = float(np.mean(np.maximum(scores, 0.0)))
        downside_pressure = float(np.mean(np.maximum(-scores, 0.0)))
        cash_logit = (
            -0.10
            + ((1.0 - risk_appetite) * 1.25)
            + (0.80 * bear)
            + (1.60 * downside_pressure)
            - (1.20 * signal_strength)
            + (0.30 * previous_cash)
        )
        return _softmax_logits(np.concatenate([asset_logits, np.array([cash_logit], dtype=float)]))


@dataclass
class MetaSignalPolicy:
    action_dim: int
    observation_dim: int
    class_feature_dim: int = 12
    class_ranges: dict[str, tuple[int, int]] | None = None
    cash_enabled: bool = True

    def predict(self, observation: np.ndarray) -> np.ndarray:
        aligned = align_observation(observation, self.observation_dim)
        n_assets = int(self.action_dim)
        slices = build_feature_slices(
            n_assets=n_assets,
            micro_dim=0,
            macro_dim=max(
                self.observation_dim
                - ((n_assets * 4) + int(self.class_feature_dim) + 2 + 3),
                0,
            ),
            class_feature_dim=int(self.class_feature_dim),
        )
        expected_returns = np.asarray(aligned[slices.expected_returns], dtype=float)
        covariance_risk = np.clip(np.asarray(aligned[slices.covariance_risk], dtype=float), 1e-10, None)
        previous_weights = np.asarray(aligned[slices.previous_weights], dtype=float)
        sub_agent_signals = np.asarray(aligned[slices.sub_agent_signals], dtype=float)
        class_context = np.asarray(aligned[slices.class_context], dtype=float)
        risk_appetite = float(np.clip(aligned[slices.portfolio_state][1], 0.0, 1.0))
        regime = np.asarray(aligned[slices.regime_features], dtype=float)
        bull = float(regime[0]) if regime.size >= 1 else 0.0
        bear = float(regime[2]) if regime.size >= 3 else 0.0

        cash_index = n_assets - 1 if self.cash_enabled and n_assets > 1 else None
        risky_slice = slice(0, cash_index) if cash_index is not None else slice(0, n_assets)
        risky_mu = expected_returns[risky_slice]
        risky_vol = np.sqrt(covariance_risk[risky_slice])
        sharpe_like = risky_mu / np.clip(risky_vol, 1e-6, None)
        prior = sub_agent_signals[risky_slice]
        prev = previous_weights[risky_slice]
        asset_scores = (
            (0.65 * prior)
            + (0.35 * sharpe_like)
            + (0.12 * prev)
            + (0.20 * bull * np.clip(risky_mu, 0.0, None))
            - (0.25 * bear * risky_vol)
        )

        ranges = self.class_ranges or {}
        class_width = 4 if int(self.class_feature_dim) >= 12 else 3
        if ranges:
            ordered_ranges = [
                (asset_class, tuple(int(value) for value in values))
                for asset_class, values in ranges.items()
                if asset_class != "cash"
            ]
            ordered_ranges.sort(key=lambda item: item[1][0])
            for index, (_, (start, end)) in enumerate(ordered_ranges):
                offset = index * class_width
                if offset + 2 >= class_context.shape[0]:
                    break
                class_expected = float(class_context[offset])
                class_volatility = float(class_context[offset + 1])
                class_prev = float(class_context[offset + 2])
                class_cash = float(class_context[offset + 3]) if class_width >= 4 else 0.0
                class_bonus = (
                    (0.45 * class_expected)
                    - (0.30 * class_volatility)
                    - (0.12 * class_prev)
                    - (0.20 * class_cash)
                )
                asset_scores[start:end] += class_bonus

        risky_logits = asset_scores * (1.3 + (1.5 * risk_appetite))
        if cash_index is None:
            return _softmax_logits(risky_logits)

        positive_share = float(np.mean(risky_mu > 0.0))
        signal_strength = float(np.mean(np.maximum(sharpe_like, 0.0)))
        cash_prior = float(sub_agent_signals[cash_index])
        cash_logit = (
            -0.20
            + ((1.0 - risk_appetite) * 1.45)
            + (0.90 * cash_prior)
            + (0.70 * bear)
            + max(0.55 - positive_share, 0.0)
            - (0.85 * signal_strength)
        )
        return _softmax_logits(np.concatenate([risky_logits, np.array([cash_logit], dtype=float)]))


@dataclass
class SB3Policy:
    model: object
    observation_dim: int

    def predict(self, observation: np.ndarray) -> np.ndarray:
        aligned = align_observation(observation, self.observation_dim).reshape(1, -1)
        action, _ = self.model.predict(aligned, deterministic=True)
        return normalize_weights(np.asarray(action, dtype=float))


def _build_box_spaces(
    *, observation_dim: int | None, action_dim: int | None
) -> tuple[spaces.Box, spaces.Box] | tuple[None, None]:
    if observation_dim is None or action_dim is None:
        return None, None
    if spaces is None:
        raise PolicyLoadError("gymnasium is not installed")
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(int(observation_dim),),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(int(action_dim),),
        dtype=np.float32,
    )
    return observation_space, action_space


def _load_sb3_policy(
    model_path: Path,
    algorithm: str,
    *,
    observation_dim: int | None = None,
    action_dim: int | None = None,
) -> SB3Policy:
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:  # pragma: no cover - exercised by health checks
        raise PolicyLoadError("stable-baselines3 is not installed") from exc

    install_numpy_pickle_compat()
    observation_space, action_space = _build_box_spaces(
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
    custom_objects = None
    if observation_space is not None and action_space is not None:
        custom_objects = {
            "observation_space": observation_space,
            "action_space": action_space,
        }

    algorithm = algorithm.lower()
    if algorithm == "ppo":
        model = PPO.load(model_path, custom_objects=custom_objects)
    elif algorithm == "sac":
        model = SAC.load(model_path, custom_objects=custom_objects)
    else:  # pragma: no cover - configuration error
        raise PolicyLoadError(f"Unsupported SB3 algorithm: {algorithm}")

    observation_dim = int(np.prod(model.observation_space.shape))
    return SB3Policy(model=model, observation_dim=observation_dim)


def load_policy(
    model_path: Path,
    metadata: dict,
    *,
    observation_dim: int | None = None,
    action_dim: int | None = None,
) -> FixedWeightPolicy | LinearPolicy | SingleAgentSignalPolicy | MetaSignalPolicy | SB3Policy:
    backend = metadata.get("policy_backend", "sb3").lower()
    if backend == "fixed":
        payload = json.loads(model_path.read_text())
        return FixedWeightPolicy(
            weights=np.asarray(payload["weights"], dtype=float),
            observation_dim=payload.get("observation_dim"),
        )
    if backend == "linear":
        payload = json.loads(model_path.read_text())
        matrix = np.asarray(payload["weights_matrix"], dtype=float)
        bias = payload.get("bias")
        return LinearPolicy(
            weights_matrix=matrix,
            bias=np.asarray(
                bias if bias is not None else np.zeros(matrix.shape[0], dtype=float),
                dtype=float,
            ),
            observation_dim=int(payload["observation_dim"]),
        )
    if backend == "single_agent_signal":
        payload = json.loads(model_path.read_text()) if model_path.exists() else {}
        resolved_action_dim = int(payload.get("action_dim", metadata.get("action_dim", action_dim)))
        resolved_observation_dim = int(
            payload.get(
                "observation_dim",
                observation_dim
                or metadata.get("inference_observation_dim")
                or metadata.get("policy_observation_dim"),
            )
        )
        return SingleAgentSignalPolicy(
            action_dim=resolved_action_dim,
            observation_dim=resolved_observation_dim,
            cash_enabled=bool(payload.get("cash_enabled", metadata.get("cash_enabled", True))),
        )
    if backend == "meta_signal":
        payload = json.loads(model_path.read_text()) if model_path.exists() else {}
        resolved_action_dim = int(payload.get("action_dim", metadata.get("action_dim", action_dim)))
        resolved_observation_dim = int(
            payload.get(
                "observation_dim",
                observation_dim
                or metadata.get("inference_observation_dim")
                or metadata.get("policy_observation_dim"),
            )
        )
        class_ranges = payload.get("class_ranges", metadata.get("class_ranges"))
        resolved_ranges = None
        if class_ranges:
            resolved_ranges = {
                key: tuple(int(value) for value in values)
                for key, values in class_ranges.items()
            }
        return MetaSignalPolicy(
            action_dim=resolved_action_dim,
            observation_dim=resolved_observation_dim,
            class_feature_dim=int(
                payload.get(
                    "class_feature_dim",
                    metadata.get("class_feature_dim", 12),
                )
            ),
            class_ranges=resolved_ranges,
            cash_enabled=bool(payload.get("cash_enabled", metadata.get("cash_enabled", True))),
        )

    return _load_sb3_policy(
        model_path,
        metadata.get("algorithm", "ppo"),
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
