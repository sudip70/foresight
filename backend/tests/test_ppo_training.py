from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from backend.app.ml.policies import (
    apply_cash_risk_off_overlay,
    apply_class_guardrails,
    normalize_weights,
    normalize_weights_with_caps,
)
from offline.ppo_training import (
    PPOPortfolioEnv,
    PPOTickerActionEnv,
    PPOTrainingConfig,
    train_asset_agent,
)


def _write_training_artifacts(asset_dir: Path) -> None:
    rows = 220
    assets = 3
    dates = np.arange(rows)
    trend = np.linspace(0.0, 15.0, rows)
    prices = np.column_stack(
        [
            100 + trend + np.sin(dates / 9),
            90 + trend * 0.8 + np.cos(dates / 11),
            110 + trend * 1.2 + np.sin(dates / 7),
        ]
    )
    regimes = np.where(np.sin(dates / 20) > 0.3, 0, np.where(np.sin(dates / 20) < -0.3, 2, 1))
    micro = np.column_stack(
        [np.sin(dates / (i + 5)) for i in range(29)] + [np.ones(rows)]
    )
    macro = np.column_stack(
        [np.cos(dates / (i + 7)) for i in range(5)] + [np.ones(rows)]
    )

    asset_dir.mkdir(parents=True, exist_ok=True)
    np.save(asset_dir / "prices.npy", prices)
    np.save(asset_dir / "regimes.npy", regimes)
    np.save(asset_dir / "micro_indicators.npy", micro)
    np.save(asset_dir / "macro_indicators.npy", macro)
    (asset_dir / "tickers.json").write_text(json.dumps(["AAA", "BBB", "CCC"]) + "\n")
    (asset_dir / "metadata.json").write_text(
        json.dumps({"asset_class": "stock", "feature_version": "ohlcv-v2"}, indent=2) + "\n"
    )


def test_ppo_portfolio_env_matches_expected_observation_shape() -> None:
    rows = 140
    prices = np.column_stack(
        [
            np.linspace(100.0, 120.0, rows),
            np.linspace(90.0, 118.0, rows),
            np.linspace(110.0, 135.0, rows),
        ]
    )
    ohlcv = np.repeat(prices[:, :, None], 5, axis=2)
    ohlcv[:, :, 4] = 1_000_000.0
    regimes = np.zeros(rows, dtype=int)
    micro = np.zeros((rows, 30), dtype=float)
    macro = np.zeros((rows, 6), dtype=float)

    env = PPOPortfolioEnv(
        prices=prices,
        ohlcv=ohlcv,
        regimes=regimes,
        micro_indicators=micro,
        macro_indicators=macro,
        window_size=10,
        episode_length=30,
        risk_low=0.2,
        risk_high=0.8,
        transaction_fee=0.001,
        turnover_penalty=0.001,
        variance_penalty_scale=1.0,
        concentration_penalty_scale=0.01,
        drawdown_penalty_scale=0.02,
        benchmark_reward_scale=0.25,
        target_daily_volatility=0.01,
        target_volatility_penalty_scale=0.25,
        max_asset_weight=0.45,
        random_start=False,
        fixed_risk=0.5,
    )
    observation, _ = env.reset()

    assert observation.shape == env.observation_space.shape
    next_observation, reward, terminated, truncated, info = env.step(np.array([0.4, 0.3, 0.3]))
    assert next_observation.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert "portfolio_value" in info
    assert "raw_reward" in info
    assert "portfolio_volatility" in info
    assert "target_volatility_penalty" in info
    assert "concentration_excess" in info


def _build_ticker_action_env() -> PPOTickerActionEnv:
    rows = 140
    risky_prices = np.column_stack(
        [
            np.linspace(100.0, 120.0, rows),
            np.linspace(90.0, 118.0, rows),
            np.linspace(110.0, 135.0, rows),
        ]
    )
    cash_prices = np.ones((rows, 1), dtype=float)
    prices = np.hstack([risky_prices, cash_prices])
    ohlcv = np.repeat(prices[:, :, None], 5, axis=2)
    ohlcv[:, :, 4] = 1_000_000.0
    regimes = np.zeros(rows, dtype=int)
    micro = np.zeros((rows, 30), dtype=float)
    macro = np.zeros((rows, 6), dtype=float)

    return PPOTickerActionEnv(
        prices=prices,
        ohlcv=ohlcv,
        regimes=regimes,
        micro_indicators=micro,
        macro_indicators=macro,
        window_size=10,
        episode_length=30,
        risk_low=0.2,
        risk_high=0.8,
        transaction_fee=0.001,
        turnover_penalty=0.001,
        variance_penalty_scale=1.0,
        concentration_penalty_scale=0.01,
        drawdown_penalty_scale=0.02,
        benchmark_reward_scale=0.25,
        target_daily_volatility=0.01,
        target_volatility_penalty_scale=0.25,
        max_asset_weight=0.45,
        max_cash_weight=0.95,
        random_start=False,
        fixed_risk=0.5,
        risky_asset_count=3,
        tradable_asset_count=3,
    )


def test_ticker_action_env_uses_one_buy_sell_hold_decision_per_ticker() -> None:
    env = _build_ticker_action_env()
    observation, _ = env.reset()

    assert observation.shape == env.observation_space.shape
    assert env.action_space.nvec.tolist() == [3, 5, 3, 5, 3, 5]
    assert np.allclose(env.asset_weights, np.array([0.0, 0.0, 0.0, 1.0]))

    buy_first_ticker = np.array(
        [
            env.ACTION_BUY,
            3,
            env.ACTION_HOLD,
            0,
            env.ACTION_HOLD,
            0,
        ]
    )
    _, reward, terminated, truncated, info = env.step(buy_first_ticker)

    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["action_layout"] == "per_ticker_buy_sell_hold"
    assert info["invalid_action_count"] == 0
    assert len(info["ticker_actions"]) == 1
    assert info["ticker_actions"][0]["action"] == "buy"
    assert np.isclose(env.asset_weights[0], 0.10)
    assert np.isclose(env.asset_weights[3], 0.90)


def test_ticker_action_env_can_trade_multiple_tickers_once_each_step() -> None:
    env = _build_ticker_action_env()
    env.reset()
    env.asset_weights = np.array([0.20, 0.20, 0.0, 0.60], dtype=float)

    buy_one_sell_one = np.array(
        [
            env.ACTION_BUY,
            2,
            env.ACTION_SELL,
            3,
            env.ACTION_HOLD,
            0,
        ]
    )
    _, _, _, _, info = env.step(buy_one_sell_one)

    assert info["invalid_action_count"] == 0
    assert [entry["action"] for entry in info["ticker_actions"]] == ["sell", "buy"]
    assert np.isclose(env.asset_weights[0], 0.25)
    assert np.isclose(env.asset_weights[1], 0.10)
    assert np.isclose(env.asset_weights[2], 0.0)
    assert np.isclose(env.asset_weights[3], 0.65)


def test_normalize_weights_respects_max_weight_cap() -> None:
    weights = normalize_weights(np.array([10.0, 1.0, 1.0]), max_weight=0.45)

    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.45 + 1e-12


def test_normalize_weights_with_caps_allows_cash_sleeve() -> None:
    weights = normalize_weights_with_caps(
        np.array([0.80, 0.10, 0.10]),
        np.array([0.20, 0.20, 0.95]),
    )

    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] <= 0.20 + 1e-12
    assert weights[1] <= 0.20 + 1e-12
    assert weights[2] > 0.50


def test_cash_risk_off_overlay_de_risks_negative_expected_return() -> None:
    weights, adjustment = apply_cash_risk_off_overlay(
        np.array([0.50, 0.49, 0.01]),
        np.array([-0.001, -0.001, 0.00015]),
        cash_index=2,
        max_cash_weight=0.95,
        target_return=0.0,
    )

    assert adjustment is not None
    assert np.isclose(weights.sum(), 1.0)
    assert weights[2] > 0.80
    assert float(np.dot(weights, np.array([-0.001, -0.001, 0.00015]))) >= -1e-12


def test_apply_class_guardrails_caps_crypto_without_stock_floor() -> None:
    weights = apply_class_guardrails(
        np.array([0.05, 0.05, 0.80, 0.10]),
        class_ranges={"stock": (0, 2), "crypto": (2, 3), "etf": (3, 4)},
        max_class_weights={"stock": 0.85, "crypto": 0.30, "etf": 0.70},
        max_asset_weight=0.70,
    )

    assert np.isclose(weights.sum(), 1.0)
    assert weights[2] <= 0.30 + 1e-12
    assert weights[:2].sum() < 0.50


def test_train_asset_agent_saves_model_and_training_summary(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    asset_dir = artifact_root / "stock"
    _write_training_artifacts(asset_dir)

    report = train_asset_agent(
        asset_class="stock",
        artifact_root=artifact_root,
        config=PPOTrainingConfig(
            total_timesteps=64,
            eval_freq=32,
            episode_length=32,
            window_size=10,
            eval_ratio=0.2,
            n_steps=32,
            batch_size=32,
            eval_episodes=1,
            device="cpu",
        ),
    )

    assert Path(report.model_path).exists()
    assert (asset_dir / "training_summary.json").exists()
    assert (asset_dir / "feature_selection.json").exists()
    assert (asset_dir / "indicator_scaler.pkl").exists()
    assert (asset_dir / "macro_scaler.pkl").exists()

    metadata = json.loads((asset_dir / "metadata.json").read_text())
    assert metadata["algorithm"] == "ppo"
    assert metadata["policy_backend"] == "sb3"
    assert metadata["feature_version"] == "ohlcv-stationary-v4-cash-sleeve"
    assert metadata["feature_selection"]["micro_selected_count"] == 29
    assert metadata["feature_selection"]["macro_selected_count"] == 5
    assert metadata["action_dim"] == 4
    assert metadata["sub_agent_architecture"]["cash_sleeve_enabled"] is True
    assert metadata["sub_agent_architecture"]["objective"] == "horizon_return_plus_diversification"
    assert metadata["eval_mean_final_value"] > 0
    assert metadata["eval_mean_sharpe"] == report.eval_mean_sharpe
