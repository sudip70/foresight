from __future__ import annotations

import numpy as np

from backend.app.ml.envs import meta_observation_dim, single_agent_observation_dim
from backend.app.ml.policies import (
    MetaSignalPolicy,
    SingleAgentSignalPolicy,
    apply_cash_risk_managed_overlay,
    blend_allocation_sources,
    constrain_turnover,
    policy_signal_blend_weight,
)


def _single_agent_observation(
    *,
    risk: float,
    bull: float,
    bear: float,
    risky_returns: tuple[float, float],
) -> np.ndarray:
    obs_dim = single_agent_observation_dim(n_assets=3, micro_dim=0, macro_dim=0)
    observation = np.zeros(obs_dim, dtype=np.float32)
    cash_return = 0.00015
    blocks = [
        [risky_returns[0] * 0.6, risky_returns[1] * 0.6, cash_return],
        [risky_returns[0] * 0.9, risky_returns[1] * 0.9, cash_return],
        [risky_returns[0], risky_returns[1], cash_return],
        [risky_returns[0] * 1.1, risky_returns[1] * 1.1, cash_return],
        [0.05, 0.08, 0.0],
        [0.04, 0.06, 0.0],
        [0.02, 0.03, 0.0],
        [risky_returns[0] * 0.2, risky_returns[1] * 0.2, cash_return],
        [0.10, 0.08, 0.0],
        [-0.14 if risky_returns[0] >= 0 else -0.32, -0.12 if risky_returns[1] >= 0 else -0.28, 0.0],
        [risky_returns[0] * 0.8, risky_returns[1] * 0.8, cash_return],
    ]
    cursor = 0
    for block in blocks:
        observation[cursor : cursor + 3] = np.asarray(block, dtype=np.float32)
        cursor += 3
    observation[cursor : cursor + 3] = np.asarray([0.35, 0.35, 0.30], dtype=np.float32)
    cursor += 3
    observation[cursor] = 1.0
    observation[cursor + 1] = float(risk)
    cursor += 2
    observation[cursor : cursor + 3] = np.asarray([bull, 1.0 - bull - bear, bear], dtype=np.float32)
    return observation


def _meta_observation(
    *,
    risk: float,
    bull: float,
    bear: float,
    positive: bool,
) -> np.ndarray:
    obs_dim = meta_observation_dim(n_assets=7, micro_dim=0, macro_dim=0, class_feature_dim=12)
    observation = np.zeros(obs_dim, dtype=np.float32)
    expected = np.asarray(
        [0.0022, 0.0019, 0.0016, 0.0013, 0.0014, 0.0011, 0.00015]
        if positive
        else [-0.0018, -0.0015, -0.0014, -0.0011, -0.0012, -0.0009, 0.00015],
        dtype=np.float32,
    )
    covariance = np.asarray([0.00025, 0.00028, 0.00034, 0.00031, 0.00022, 0.00020, 1e-8], dtype=np.float32)
    previous = np.asarray([0.14, 0.12, 0.12, 0.10, 0.14, 0.13, 0.25], dtype=np.float32)
    sub_agent = np.asarray([0.17, 0.13, 0.12, 0.10, 0.14, 0.11, 0.23], dtype=np.float32)
    class_context = np.asarray(
        [
            0.0018 if positive else -0.0014,
            0.012,
            0.26,
            0.10,
            0.0014 if positive else -0.0011,
            0.014,
            0.22,
            0.16,
            0.0012 if positive else -0.0008,
            0.010,
            0.27,
            0.12,
        ],
        dtype=np.float32,
    )
    cursor = 0
    for block in (expected, covariance, previous, sub_agent, class_context):
        observation[cursor : cursor + block.shape[0]] = block
        cursor += block.shape[0]
    observation[cursor] = 1.0
    observation[cursor + 1] = float(risk)
    cursor += 2
    observation[cursor : cursor + 3] = np.asarray([bull, 1.0 - bull - bear, bear], dtype=np.float32)
    return observation


def test_single_agent_signal_policy_responds_to_risk_and_regime() -> None:
    policy = SingleAgentSignalPolicy(action_dim=3, observation_dim=single_agent_observation_dim(n_assets=3, micro_dim=0, macro_dim=0))
    bearish_low_risk = policy.predict(
        _single_agent_observation(risk=0.2, bull=0.0, bear=1.0, risky_returns=(-0.020, -0.015))
    )
    bullish_high_risk = policy.predict(
        _single_agent_observation(risk=0.8, bull=1.0, bear=0.0, risky_returns=(0.022, 0.018))
    )

    assert bearish_low_risk[-1] > 0.45
    assert bullish_high_risk[-1] < bearish_low_risk[-1]
    assert bullish_high_risk[:2].sum() > bearish_low_risk[:2].sum()


def test_meta_signal_policy_deploys_more_risk_when_signals_improve() -> None:
    policy = MetaSignalPolicy(
        action_dim=7,
        observation_dim=meta_observation_dim(n_assets=7, micro_dim=0, macro_dim=0, class_feature_dim=12),
        class_feature_dim=12,
        class_ranges={
            "stock": (0, 2),
            "crypto": (2, 4),
            "etf": (4, 6),
            "cash": (6, 7),
        },
        cash_enabled=True,
    )
    bearish_low_risk = policy.predict(_meta_observation(risk=0.2, bull=0.0, bear=1.0, positive=False))
    bullish_high_risk = policy.predict(_meta_observation(risk=0.8, bull=1.0, bear=0.0, positive=True))

    assert bearish_low_risk[-1] > bullish_high_risk[-1]
    assert bullish_high_risk[:6].sum() > bearish_low_risk[:6].sum()


def test_cash_risk_managed_overlay_scales_cash_smoothly() -> None:
    weights = np.asarray([0.30, 0.25, 0.25, 0.20], dtype=float)
    expected_returns = np.asarray([-0.0015, -0.0012, -0.0008, 0.00015], dtype=float)
    covariance_diag = np.asarray([0.00045, 0.00030, 0.00022, 1e-8], dtype=float)

    low_risk_weights, low_info = apply_cash_risk_managed_overlay(
        weights,
        expected_returns,
        covariance_diag,
        cash_index=3,
        max_cash_weight=0.95,
        risk_appetite=0.2,
        cash_prior=0.30,
        target_return=0.0,
    )
    high_risk_weights, high_info = apply_cash_risk_managed_overlay(
        weights,
        expected_returns,
        covariance_diag,
        cash_index=3,
        max_cash_weight=0.95,
        risk_appetite=0.8,
        cash_prior=0.10,
        target_return=0.0,
    )

    assert low_info is not None
    assert high_info is not None
    assert low_risk_weights[-1] > high_risk_weights[-1] > weights[-1]
    assert low_info["cash_target"] > high_info["cash_target"]


def test_policy_signal_blend_weight_increases_for_weaker_models() -> None:
    strong_model = policy_signal_blend_weight(
        {
            "eval_mean_benchmark_alpha": 0.0010,
            "eval_mean_sharpe": 1.6,
            "eval_mean_final_value": 1.15,
        }
    )
    weak_model = policy_signal_blend_weight(
        {
            "eval_mean_benchmark_alpha": -0.0003,
            "eval_mean_sharpe": 0.4,
            "eval_mean_final_value": 0.97,
        }
    )

    assert weak_model > strong_model
    assert strong_model >= 0.0
    assert weak_model <= 0.85


def test_blend_allocation_sources_moves_toward_secondary_source() -> None:
    meta_weights = np.asarray([0.70, 0.20, 0.10], dtype=float)
    agent_weights = np.asarray([0.10, 0.70, 0.20], dtype=float)

    mixed = blend_allocation_sources(
        meta_weights,
        agent_weights,
        secondary_weight=0.50,
    )

    assert np.isclose(mixed.sum(), 1.0)
    assert mixed[1] > meta_weights[1]
    assert mixed[0] < meta_weights[0]


def test_constrain_turnover_limits_rebalance_size() -> None:
    previous = np.asarray([0.70, 0.20, 0.10], dtype=float)
    target = np.asarray([0.10, 0.20, 0.70], dtype=float)

    constrained = constrain_turnover(target, previous, max_turnover=0.20)
    turnover = float(np.sum(np.abs(constrained - previous)))

    assert turnover <= 0.2000001
    assert constrained[0] > target[0]
    assert constrained[2] < target[2]
