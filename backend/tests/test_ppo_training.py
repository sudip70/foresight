from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from backend.app.ml.policies import apply_class_guardrails, normalize_weights
from offline.ppo_training import PPOPortfolioEnv, PPOTrainingConfig, train_asset_agent


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


def test_normalize_weights_respects_max_weight_cap() -> None:
    weights = normalize_weights(np.array([10.0, 1.0, 1.0]), max_weight=0.45)

    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.45 + 1e-12


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
    assert metadata["feature_version"] == "ohlcv-stationary-v3"
    assert metadata["feature_selection"]["micro_selected_count"] == 29
    assert metadata["feature_selection"]["macro_selected_count"] == 5
    assert metadata["eval_mean_final_value"] > 0
    assert metadata["eval_mean_sharpe"] == report.eval_mean_sharpe
