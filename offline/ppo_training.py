from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
import shutil
import tempfile

import gymnasium as gym
from gymnasium import spaces
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.app.ml.envs import build_single_agent_observation, single_agent_observation_dim
from backend.app.ml.policies import normalize_action_with_cash_sleeve


@dataclass(frozen=True)
class PPOTrainingConfig:
    total_timesteps: int = 25_000
    eval_freq: int = 2_500
    episode_length: int = 252
    window_size: int = 60
    eval_ratio: float = 0.2
    risk_low: float = 0.2
    risk_high: float = 0.8
    transaction_fee: float = 0.001
    turnover_penalty: float = 0.001
    variance_penalty_scale: float = 1.0
    concentration_penalty_scale: float = 0.01
    drawdown_penalty_scale: float = 0.02
    downside_penalty_scale: float = 0.25
    benchmark_reward_scale: float = 0.25
    horizon_reward_scale: float = 1.0
    diversification_reward_scale: float = 0.10
    target_daily_volatility: float | None = None
    target_volatility_penalty_scale: float = 0.0
    max_asset_weight: float | None = None
    max_cash_weight: float | None = 0.95
    cash_enabled: bool = True
    cash_annual_return: float = 0.04
    reward_scale: float = 100.0
    learning_rate: float = 3e-4
    learning_rate_schedule: str = "constant"
    learning_rate_final: float | None = None
    n_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_schedule: str = "constant"
    clip_range_final: float | None = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    normalize_advantage: bool = True
    use_sde: bool = False
    sde_sample_freq: int = -1
    policy_layers: tuple[int, ...] = (128, 128)
    activation_fn: str = "tanh"
    orthogonal_init: bool = True
    target_kl: float | None = 0.03
    max_grad_norm: float = 0.5
    n_envs: int = 1
    min_feature_variance: float = 1e-10
    feature_clip: float = 5.0
    random_start: bool = True
    eval_episodes: int = 3
    seed: int = 42
    device: str = "cpu"


@dataclass(frozen=True)
class PPOTrainingReport:
    asset_class: str
    total_timesteps: int
    train_rows: int
    eval_rows: int
    eval_mean_reward: float
    eval_mean_final_value: float
    eval_mean_sharpe: float
    eval_mean_max_drawdown: float
    eval_mean_benchmark_alpha: float
    model_path: str
    backup_path: str | None
    trained_at: str


@dataclass(frozen=True)
class FeaturePreparationReport:
    micro_source: str
    macro_source: str
    micro_original_count: int
    macro_original_count: int
    micro_selected_count: int
    macro_selected_count: int
    micro_dropped_indices: list[int]
    macro_dropped_indices: list[int]


def _cash_daily_return(config: PPOTrainingConfig) -> float:
    return (1.0 + float(config.cash_annual_return)) ** (1.0 / 252.0) - 1.0


def _risk_cash_cap(risk: float, *, configured_max: float | None) -> float | None:
    if configured_max is None:
        return None
    risk_value = float(np.clip(risk, 0.0, 1.0))
    dynamic_cap = 0.08 + ((1.0 - risk_value) * 0.42)
    return float(min(float(configured_max), dynamic_cap))


def _risk_concentration_target(n_assets: int, risk: float) -> float:
    risk_value = float(np.clip(risk, 0.0, 1.0))
    equal_concentration = 1.0 / max(int(n_assets), 1)
    return equal_concentration + (risk_value * (0.45 - equal_concentration))


def _append_cash_sleeve_arrays(
    prices: np.ndarray,
    ohlcv: np.ndarray,
    *,
    config: PPOTrainingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if not config.cash_enabled:
        return np.asarray(prices, dtype=float), np.asarray(ohlcv, dtype=float)

    base_prices = np.asarray(prices, dtype=float)
    base_ohlcv = np.asarray(ohlcv, dtype=float)
    cash_prices = np.cumprod(
        np.full(base_prices.shape[0], 1.0 + _cash_daily_return(config), dtype=float)
    ).reshape(-1, 1)
    cash_ohlcv = np.zeros((base_prices.shape[0], 1, 5), dtype=float)
    cash_ohlcv[:, 0, 0] = cash_prices[:, 0]
    cash_ohlcv[:, 0, 1] = cash_prices[:, 0]
    cash_ohlcv[:, 0, 2] = cash_prices[:, 0]
    cash_ohlcv[:, 0, 3] = cash_prices[:, 0]
    cash_ohlcv[:, 0, 4] = 1.0
    return np.hstack([base_prices, cash_prices]), np.concatenate([base_ohlcv, cash_ohlcv], axis=1)


def _load_json(path: Path, fallback: dict | None = None) -> dict:
    if not path.exists():
        return {} if fallback is None else dict(fallback)
    data = json.loads(path.read_text())
    if fallback is None:
        return data
    merged = dict(fallback)
    merged.update(data)
    return merged


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_feature_names(asset_dir: Path, key: str, width: int) -> list[str]:
    path = asset_dir / "feature_names.json"
    if not path.exists():
        return [f"{key}_{index}" for index in range(width)]
    payload = _load_json(path)
    names = payload.get(key, [])
    if len(names) != width:
        return [f"{key}_{index}" for index in range(width)]
    return list(names)


def _write_selected_feature_names(
    asset_dir: Path,
    *,
    micro_indices: np.ndarray,
    macro_indices: np.ndarray,
    micro_width: int,
    macro_width: int,
) -> None:
    path = asset_dir / "feature_names.json"
    payload = _load_json(path) if path.exists() else {}
    raw_micro = payload.get("micro_raw", payload.get("micro"))
    raw_macro = payload.get("macro_raw", payload.get("macro"))
    if raw_micro is None or len(raw_micro) != micro_width:
        raw_micro = _load_feature_names(asset_dir, "micro", micro_width)
    if raw_macro is None or len(raw_macro) != macro_width:
        raw_macro = _load_feature_names(asset_dir, "macro", macro_width)

    payload["micro_raw"] = list(raw_micro)
    payload["macro_raw"] = list(raw_macro)
    payload["micro"] = [raw_micro[index] for index in micro_indices.tolist()]
    payload["macro"] = [raw_macro[index] for index in macro_indices.tolist()]
    _save_json(path, payload)


def _select_and_scale_features(
    features: np.ndarray,
    *,
    split_index: int,
    min_feature_variance: float,
    feature_clip: float,
) -> tuple[np.ndarray, RobustScaler, np.ndarray, list[int]]:
    features = np.asarray(features, dtype=float)
    train_features = features[:split_index]
    variance = np.nanvar(train_features, axis=0)
    selected = np.flatnonzero(np.isfinite(variance) & (variance > min_feature_variance))
    if selected.size == 0:
        selected = np.arange(features.shape[1])

    filtered = np.nan_to_num(features[:, selected], nan=0.0, posinf=0.0, neginf=0.0)
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    scaler.fit(filtered[:split_index])
    scaled = scaler.transform(filtered)
    if feature_clip > 0:
        scaled = np.clip(scaled, -feature_clip, feature_clip)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    dropped = sorted(set(range(features.shape[1])) - set(selected.tolist()))
    return scaled.astype(float), scaler, selected.astype(int), dropped


def _prepare_indicator_features(
    *,
    asset_dir: Path,
    micro: np.ndarray,
    macro: np.ndarray,
    micro_source: str,
    macro_source: str,
    split_index: int,
    config: PPOTrainingConfig,
) -> tuple[np.ndarray, np.ndarray, FeaturePreparationReport]:
    micro_width = int(micro.shape[1])
    macro_width = int(macro.shape[1])
    micro_scaled, micro_scaler, micro_indices, micro_dropped = _select_and_scale_features(
        micro,
        split_index=split_index,
        min_feature_variance=config.min_feature_variance,
        feature_clip=config.feature_clip,
    )
    macro_scaled, macro_scaler, macro_indices, macro_dropped = _select_and_scale_features(
        macro,
        split_index=split_index,
        min_feature_variance=config.min_feature_variance,
        feature_clip=config.feature_clip,
    )

    np.save(asset_dir / "micro_indicators.npy", micro_scaled)
    np.save(asset_dir / "macro_indicators.npy", macro_scaled)
    joblib.dump(micro_scaler, asset_dir / "indicator_scaler.pkl")
    joblib.dump(macro_scaler, asset_dir / "macro_scaler.pkl")
    _write_selected_feature_names(
        asset_dir,
        micro_indices=micro_indices,
        macro_indices=macro_indices,
        micro_width=micro_width,
        macro_width=macro_width,
    )

    report = FeaturePreparationReport(
        micro_source=micro_source,
        macro_source=macro_source,
        micro_original_count=micro_width,
        macro_original_count=macro_width,
        micro_selected_count=int(micro_scaled.shape[1]),
        macro_selected_count=int(macro_scaled.shape[1]),
        micro_dropped_indices=micro_dropped,
        macro_dropped_indices=macro_dropped,
    )
    _save_json(asset_dir / "feature_selection.json", asdict(report))
    return micro_scaled, macro_scaled, report


def _load_asset_training_arrays(
    asset_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    tickers = json.loads((asset_dir / "tickers.json").read_text())
    prices = np.load(asset_dir / "prices.npy")
    ohlcv_path = asset_dir / "ohlcv.npy"
    ohlcv = np.load(ohlcv_path) if ohlcv_path.exists() else None
    regimes = np.load(asset_dir / "regimes.npy")
    micro_path = asset_dir / "micro_indicators_raw.npy"
    macro_path = asset_dir / "macro_indicators_raw.npy"
    micro_source_path = micro_path if micro_path.exists() else asset_dir / "micro_indicators.npy"
    macro_source_path = macro_path if macro_path.exists() else asset_dir / "macro_indicators.npy"
    micro = np.load(micro_source_path)
    macro = np.load(macro_source_path)
    row_count = min(
        prices.shape[0],
        regimes.shape[0],
        micro.shape[0],
        macro.shape[0],
        ohlcv.shape[0] if ohlcv is not None else prices.shape[0],
    )
    if row_count <= 100:
        raise ValueError(f"Not enough rows to train PPO agent from {asset_dir}")
    if ohlcv is None:
        ohlcv = np.repeat(np.asarray(prices, dtype=float)[:, :, None], 5, axis=2)
        ohlcv[:, :, 4] = 1.0
    return (
        tickers,
        np.asarray(prices, dtype=float)[-row_count:],
        np.asarray(ohlcv, dtype=float)[-row_count:],
        np.asarray(regimes, dtype=int)[-row_count:],
        np.asarray(micro, dtype=float)[-row_count:],
        np.asarray(macro, dtype=float)[-row_count:],
        {
            "micro": micro_source_path.name,
            "macro": macro_source_path.name,
        },
    )


def compute_split_index(*, rows: int, eval_ratio: float, window_size: int) -> int:
    split_index = int(rows * (1.0 - eval_ratio))
    split_index = max(split_index, window_size + 64)
    split_index = min(split_index, rows - (window_size + 64))
    if split_index <= window_size or split_index >= rows:
        raise ValueError(f"Unable to split training rows={rows} for window_size={window_size}")
    return split_index


def split_training_data(
    *,
    prices: np.ndarray,
    ohlcv: np.ndarray,
    regimes: np.ndarray,
    micro_indicators: np.ndarray,
    macro_indicators: np.ndarray,
    eval_ratio: float,
    window_size: int,
    split_index: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows = prices.shape[0]
    if split_index is None:
        split_index = compute_split_index(
            rows=rows, eval_ratio=eval_ratio, window_size=window_size
        )

    train_slice = slice(0, split_index)
    eval_slice = slice(split_index - window_size, rows)
    return (
        {
            "prices": prices[train_slice],
            "ohlcv": ohlcv[train_slice],
            "regimes": regimes[train_slice],
            "micro": micro_indicators[train_slice],
            "macro": macro_indicators[train_slice],
        },
        {
            "prices": prices[eval_slice],
            "ohlcv": ohlcv[eval_slice],
            "regimes": regimes[eval_slice],
            "micro": micro_indicators[eval_slice],
            "macro": macro_indicators[eval_slice],
        },
    )


class PPOPortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        prices: np.ndarray,
        ohlcv: np.ndarray,
        regimes: np.ndarray,
        micro_indicators: np.ndarray,
        macro_indicators: np.ndarray,
        window_size: int,
        episode_length: int,
        risk_low: float,
        risk_high: float,
        transaction_fee: float,
        turnover_penalty: float,
        variance_penalty_scale: float,
        concentration_penalty_scale: float,
        drawdown_penalty_scale: float,
        downside_penalty_scale: float = 0.25,
        benchmark_reward_scale: float = 0.25,
        horizon_reward_scale: float = 1.0,
        diversification_reward_scale: float = 0.10,
        target_daily_volatility: float | None = None,
        target_volatility_penalty_scale: float = 0.0,
        max_asset_weight: float | None = None,
        max_cash_weight: float | None = None,
        reward_scale: float = 100.0,
        random_start: bool = True,
        fixed_risk: float | None = None,
        risky_asset_count: int | None = None,
    ) -> None:
        super().__init__()
        self.prices = np.asarray(prices, dtype=float)
        self.ohlcv = np.asarray(ohlcv, dtype=float)
        self.regimes = np.asarray(regimes, dtype=int)
        self.micro_indicators = np.asarray(micro_indicators, dtype=float)
        self.macro_indicators = np.asarray(macro_indicators, dtype=float)
        self.window_size = int(window_size)
        self.episode_length = int(episode_length)
        self.risk_low = float(risk_low)
        self.risk_high = float(risk_high)
        self.transaction_fee = float(transaction_fee)
        self.turnover_penalty = float(turnover_penalty)
        self.variance_penalty_scale = float(variance_penalty_scale)
        self.concentration_penalty_scale = float(concentration_penalty_scale)
        self.drawdown_penalty_scale = float(drawdown_penalty_scale)
        self.downside_penalty_scale = float(downside_penalty_scale)
        self.benchmark_reward_scale = float(benchmark_reward_scale)
        self.horizon_reward_scale = float(horizon_reward_scale)
        self.diversification_reward_scale = float(diversification_reward_scale)
        self.target_daily_volatility = (
            None if target_daily_volatility is None else float(target_daily_volatility)
        )
        self.target_volatility_penalty_scale = float(target_volatility_penalty_scale)
        self.max_asset_weight = None if max_asset_weight is None else float(max_asset_weight)
        self.max_cash_weight = None if max_cash_weight is None else float(max_cash_weight)
        self.reward_scale = float(reward_scale)
        self.random_start = bool(random_start)
        self.fixed_risk = None if fixed_risk is None else float(fixed_risk)
        self.num_regimes = 3
        self.n_assets = int(self.prices.shape[1])
        self.risky_asset_count = int(risky_asset_count or self.n_assets)
        self.max_start = self.prices.shape[0] - self.window_size - self.episode_length - 2
        if self.max_start < self.window_size:
            raise ValueError(
                "Not enough rows for training episode: "
                f"rows={self.prices.shape[0]}, window={self.window_size}, episode={self.episode_length}"
            )

        obs_dim = single_agent_observation_dim(
            n_assets=self.n_assets,
            micro_dim=self.micro_indicators.shape[1],
            macro_dim=self.macro_indicators.shape[1],
            num_regimes=self.num_regimes,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self.current_step = self.window_size
        self.start_step = self.window_size
        self.end_step = self.window_size + self.episode_length
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.asset_weights = np.ones(self.n_assets, dtype=float) / self.n_assets
        self.risk_appetite = 0.5
        self.episode_log_return = 0.0
        self.episode_benchmark_log_return = 0.0
        self.concentration_history: list[float] = []

    def _sample_start(self) -> int:
        if not self.random_start:
            return self.window_size
        if self.max_start == self.window_size:
            return self.window_size
        return int(self.np_random.integers(self.window_size, self.max_start + 1))

    def _sample_risk(self) -> float:
        if self.fixed_risk is not None:
            return self.fixed_risk
        return float(self.np_random.uniform(self.risk_low, self.risk_high))

    def _observation(self) -> np.ndarray:
        return build_single_agent_observation(
            prices=self.prices,
            regimes=self.regimes,
            micro_indicators=self.micro_indicators,
            macro_indicators=self.macro_indicators,
            ohlcv=self.ohlcv,
            step=self.current_step,
            risk_appetite=self.risk_appetite,
            prev_weights=self.asset_weights,
            portfolio_value_ratio=self.portfolio_value,
            num_regimes=self.num_regimes,
        )

    def _window_covariance(self) -> np.ndarray:
        start = max(0, self.current_step - self.window_size)
        window_prices = np.clip(self.prices[start : self.current_step + 1], 1e-12, None)
        if window_prices.shape[0] < 2:
            return np.zeros((self.n_assets, self.n_assets), dtype=float)
        log_returns = np.diff(np.log(window_prices), axis=0)
        if log_returns.shape[0] < 2:
            return np.zeros((self.n_assets, self.n_assets), dtype=float)
        covariance = np.cov(log_returns.T)
        if covariance.ndim == 0:
            covariance = np.array([[float(covariance)]], dtype=float)
        covariance += np.eye(covariance.shape[0]) * 1e-6
        return covariance

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.start_step = self._sample_start()
        self.current_step = self.start_step
        self.end_step = min(self.start_step + self.episode_length, self.prices.shape[0] - 2)
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.asset_weights = np.ones(self.n_assets, dtype=float) / self.n_assets
        self.risk_appetite = self._sample_risk()
        self.episode_log_return = 0.0
        self.episode_benchmark_log_return = 0.0
        self.concentration_history = []
        return self._observation(), {}

    def _advance_with_weights(
        self,
        weights: np.ndarray,
        *,
        reward_adjustment: float = 0.0,
        extra_info: dict | None = None,
    ):
        weights = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, None)
        if weights.shape != (self.n_assets,):
            raise ValueError(
                f"weights must have shape ({self.n_assets},), got {weights.shape}"
            )
        total_weight = float(weights.sum())
        if total_weight <= 1e-12:
            weights = np.ones(self.n_assets, dtype=float) / self.n_assets
        else:
            weights = weights / total_weight

        turnover = float(np.sum(np.abs(weights - self.asset_weights)))
        transaction_cost = turnover * self.transaction_fee

        current_prices = np.clip(self.prices[self.current_step], 1e-12, None)
        next_prices = np.clip(self.prices[self.current_step + 1], 1e-12, None)
        asset_returns = (next_prices - current_prices) / current_prices
        portfolio_return = float(np.dot(weights, asset_returns))
        benchmark_return = float(np.mean(asset_returns))

        covariance = self._window_covariance()
        portfolio_variance = float(weights.T @ covariance @ weights)
        portfolio_volatility = float(np.sqrt(max(portfolio_variance, 0.0)))
        concentration = float(np.sum(np.square(weights)))

        previous_drawdown = 1.0 - (self.portfolio_value / max(self.peak_value, 1e-12))
        net_return = max(1.0 + portfolio_return - transaction_cost, 1e-6)
        next_portfolio_value = self.portfolio_value * net_return
        next_peak_value = max(self.peak_value, next_portfolio_value)
        drawdown = 1.0 - (next_portfolio_value / max(next_peak_value, 1e-12))
        drawdown_worsening = max(drawdown - previous_drawdown, 0.0)
        target_concentration = _risk_concentration_target(self.n_assets, self.risk_appetite)
        concentration_excess = max(concentration - target_concentration, 0.0)
        downside_return = max(-portfolio_return, 0.0)
        risk_penalty = (1.0 - self.risk_appetite) * self.variance_penalty_scale * portfolio_volatility
        target_volatility_penalty = 0.0
        if self.target_daily_volatility is not None and self.target_volatility_penalty_scale > 0:
            target_volatility_penalty = self.target_volatility_penalty_scale * abs(
                portfolio_volatility - self.target_daily_volatility
            )
        self.episode_log_return += float(np.log(net_return))
        self.episode_benchmark_log_return += float(np.log(max(1.0 + benchmark_return, 1e-6)))
        self.concentration_history.append(concentration)
        terminated = self.current_step + 1 >= self.end_step
        terminal_reward = 0.0
        if terminated:
            horizon_alpha = self.episode_log_return - self.episode_benchmark_log_return
            average_concentration = (
                float(np.mean(self.concentration_history))
                if self.concentration_history
                else concentration
            )
            terminal_reward = (
                self.horizon_reward_scale * self.episode_log_return
                + self.benchmark_reward_scale * horizon_alpha
                - self.diversification_reward_scale
                * max(average_concentration - target_concentration, 0.0)
            )

        raw_reward = (
            float(np.log(net_return))
            + (self.benchmark_reward_scale * (portfolio_return - benchmark_return))
            + terminal_reward
            - risk_penalty
            - target_volatility_penalty
            - (self.turnover_penalty * turnover)
            - (self.concentration_penalty_scale * concentration_excess)
            - (self.downside_penalty_scale * downside_return)
            - (self.drawdown_penalty_scale * (drawdown + drawdown_worsening))
            + float(reward_adjustment)
        )
        reward = self.reward_scale * raw_reward

        self.portfolio_value = next_portfolio_value
        self.peak_value = next_peak_value
        self.asset_weights = weights
        self.current_step += 1

        observation = self._observation()
        info = {
            "portfolio_value": float(self.portfolio_value),
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "benchmark_alpha": portfolio_return - benchmark_return,
            "portfolio_variance": portfolio_variance,
            "portfolio_volatility": portfolio_volatility,
            "target_daily_volatility": self.target_daily_volatility,
            "target_volatility_penalty": target_volatility_penalty,
            "concentration": concentration,
            "concentration_excess": concentration_excess,
            "risk_penalty": risk_penalty,
            "raw_reward": raw_reward,
            "terminal_reward": terminal_reward,
            "horizon_log_return": self.episode_log_return,
            "horizon_benchmark_log_return": self.episode_benchmark_log_return,
            "drawdown": drawdown,
            "drawdown_worsening": drawdown_worsening,
            "risk_appetite": float(self.risk_appetite),
        }
        if extra_info:
            info.update(extra_info)
        return observation, reward, terminated, False, info

    def step(self, action):
        weights = normalize_action_with_cash_sleeve(
            np.asarray(action, dtype=float),
            risky_asset_count=self.risky_asset_count,
            max_risky_weight=self.max_asset_weight,
            max_cash_weight=_risk_cash_cap(self.risk_appetite, configured_max=self.max_cash_weight),
        )
        return self._advance_with_weights(
            weights,
            extra_info={"action_layout": "target_weights"},
        )


class PPOTickerActionEnv(PPOPortfolioEnv):
    """PPO environment with one buy/sell/hold decision per ticker per step."""

    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2
    ACTION_NAMES = {
        ACTION_HOLD: "hold",
        ACTION_BUY: "buy",
        ACTION_SELL: "sell",
    }
    DEFAULT_TRADE_SIZE_BUCKETS = (0.01, 0.025, 0.05, 0.10, 0.20)

    def __init__(
        self,
        *,
        trade_size_buckets: tuple[float, ...] | list[float] | np.ndarray | None = None,
        tradable_asset_count: int | None = None,
        invalid_action_penalty: float = 0.001,
        initial_cash_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tradable_asset_count = int(tradable_asset_count or self.risky_asset_count)
        if self.tradable_asset_count <= 0:
            raise ValueError("tradable_asset_count must be positive")
        if self.tradable_asset_count >= self.n_assets:
            raise ValueError("PPOTickerActionEnv requires a cash sleeve after tradable assets")

        self.cash_index = self.tradable_asset_count
        buckets = (
            self.DEFAULT_TRADE_SIZE_BUCKETS
            if trade_size_buckets is None
            else tuple(float(value) for value in trade_size_buckets)
        )
        self.trade_size_buckets = np.asarray(sorted(set(buckets)), dtype=float)
        if self.trade_size_buckets.ndim != 1 or self.trade_size_buckets.size == 0:
            raise ValueError("trade_size_buckets must contain at least one size")
        if not np.all(np.isfinite(self.trade_size_buckets)):
            raise ValueError("trade_size_buckets must be finite")
        if np.any(self.trade_size_buckets <= 0.0):
            raise ValueError("trade_size_buckets must be positive")

        self.invalid_action_penalty = float(invalid_action_penalty)
        self.initial_cash_weight = float(np.clip(initial_cash_weight, 0.0, 1.0))
        nvec = np.tile(
            np.array([3, self.trade_size_buckets.size], dtype=np.int64),
            self.tradable_asset_count,
        )
        self.action_space = spaces.MultiDiscrete(nvec)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _, info = super().reset(seed=seed, options=options)
        self.asset_weights = np.zeros(self.n_assets, dtype=float)
        residual_risky_weight = max(1.0 - self.initial_cash_weight, 0.0)
        if residual_risky_weight > 0.0:
            self.asset_weights[: self.tradable_asset_count] = (
                residual_risky_weight / self.tradable_asset_count
            )
        self.asset_weights[self.cash_index] = self.initial_cash_weight
        return self._observation(), info

    def _action_pairs(self, action) -> np.ndarray:
        raw_action = np.asarray(action, dtype=int).reshape(-1)
        expected_size = self.tradable_asset_count * 2
        if raw_action.size != expected_size:
            raise ValueError(
                f"ticker action must contain {expected_size} values, got {raw_action.size}"
            )
        pairs = raw_action.reshape(self.tradable_asset_count, 2)
        pairs[:, 0] = np.clip(pairs[:, 0], 0, 2)
        pairs[:, 1] = np.clip(pairs[:, 1], 0, self.trade_size_buckets.size - 1)
        return pairs

    def _apply_ticker_actions(self, action) -> tuple[np.ndarray, list[dict], int]:
        pairs = self._action_pairs(action)
        next_weights = self.asset_weights.copy()
        cash_weight = float(next_weights[self.cash_index])
        executed_actions: list[dict] = []
        invalid_actions = 0

        for asset_index, (action_type, size_index) in enumerate(pairs):
            if int(action_type) != self.ACTION_SELL:
                continue
            requested_size = float(self.trade_size_buckets[int(size_index)])
            weight_before = float(next_weights[asset_index])
            executable_size = min(requested_size, weight_before)
            if executable_size <= 1e-12:
                invalid_actions += 1
                continue
            next_weights[asset_index] -= executable_size
            cash_weight += executable_size
            next_weights[self.cash_index] = cash_weight
            executed_actions.append(
                {
                    "asset_index": int(asset_index),
                    "action": "sell",
                    "size_bucket": requested_size,
                    "executed_weight_delta": float(-executable_size),
                    "weight_before": weight_before,
                    "weight_after": float(next_weights[asset_index]),
                }
            )

        for asset_index, (action_type, size_index) in enumerate(pairs):
            if int(action_type) != self.ACTION_BUY:
                continue
            requested_size = float(self.trade_size_buckets[int(size_index)])
            weight_before = float(next_weights[asset_index])
            asset_cap = 1.0 if self.max_asset_weight is None else float(self.max_asset_weight)
            headroom = max(asset_cap - weight_before, 0.0)
            executable_size = min(requested_size, cash_weight, headroom)
            if executable_size <= 1e-12:
                invalid_actions += 1
                continue
            next_weights[asset_index] += executable_size
            cash_weight -= executable_size
            next_weights[self.cash_index] = cash_weight
            executed_actions.append(
                {
                    "asset_index": int(asset_index),
                    "action": "buy",
                    "size_bucket": requested_size,
                    "executed_weight_delta": float(executable_size),
                    "weight_before": weight_before,
                    "weight_after": float(next_weights[asset_index]),
                }
            )

        return next_weights, executed_actions, invalid_actions

    def step(self, action):
        weights, executed_actions, invalid_actions = self._apply_ticker_actions(action)
        reward_adjustment = -self.invalid_action_penalty * float(invalid_actions)
        return self._advance_with_weights(
            weights,
            reward_adjustment=reward_adjustment,
            extra_info={
                "action_layout": "per_ticker_buy_sell_hold",
                "ticker_actions": executed_actions,
                "invalid_action_count": int(invalid_actions),
                "invalid_action_penalty": float(-reward_adjustment),
                "cash_weight": float(weights[self.cash_index]),
            },
        )


def _evaluate_model(
    model: PPO,
    *,
    env_factory,
    episodes: int,
) -> dict[str, float]:
    rewards = []
    final_values = []
    sharpes = []
    max_drawdowns = []
    benchmark_alphas = []
    for _ in range(episodes):
        env = env_factory()
        observation, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        daily_returns: list[float] = []
        drawdowns: list[float] = []
        alphas: list[float] = []
        info = {"portfolio_value": 1.0}
        while not done and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += float(reward)
            daily_returns.append(float(info["portfolio_return"]))
            drawdowns.append(float(info["drawdown"]))
            alphas.append(float(info["benchmark_alpha"]))
        rewards.append(episode_reward)
        final_values.append(float(info["portfolio_value"]))
        volatility = float(np.std(daily_returns))
        sharpe = (
            float((np.mean(daily_returns) / volatility) * np.sqrt(252))
            if volatility > 0
            else 0.0
        )
        sharpes.append(sharpe)
        max_drawdowns.append(float(max(drawdowns) if drawdowns else 0.0))
        benchmark_alphas.append(float(np.mean(alphas) if alphas else 0.0))
    return {
        "eval_mean_reward": float(np.mean(rewards)),
        "eval_mean_final_value": float(np.mean(final_values)),
        "eval_mean_sharpe": float(np.mean(sharpes)),
        "eval_mean_max_drawdown": float(np.mean(max_drawdowns)),
        "eval_mean_benchmark_alpha": float(np.mean(benchmark_alphas)),
    }


def _activation_class(name: str):
    from torch import nn

    lookup = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    key = name.lower().replace("-", "_")
    if key not in lookup:
        raise ValueError(f"Unsupported activation_fn: {name}")
    return lookup[key]


def _scheduled_value(
    *,
    initial: float,
    final: float | None,
    schedule: str,
):
    schedule = schedule.lower()
    if schedule == "constant":
        return float(initial)
    if schedule != "linear":
        raise ValueError(f"Unsupported schedule: {schedule}")

    final_value = float(initial) * 0.1 if final is None else float(final)

    def _linear(progress_remaining: float) -> float:
        return final_value + ((float(initial) - final_value) * float(progress_remaining))

    return _linear


def train_asset_agent(
    *,
    asset_class: str,
    artifact_root: Path,
    config: PPOTrainingConfig,
) -> PPOTrainingReport:
    asset_dir = artifact_root / asset_class
    tickers, prices, ohlcv, regimes, micro, macro, feature_sources = _load_asset_training_arrays(
        asset_dir
    )
    split_index = compute_split_index(
        rows=prices.shape[0],
        eval_ratio=config.eval_ratio,
        window_size=config.window_size,
    )
    micro, macro, feature_report = _prepare_indicator_features(
        asset_dir=asset_dir,
        micro=micro,
        macro=macro,
        micro_source=feature_sources["micro"],
        macro_source=feature_sources["macro"],
        split_index=split_index,
        config=config,
    )
    risky_asset_count = int(len(tickers))
    prices, ohlcv = _append_cash_sleeve_arrays(prices, ohlcv, config=config)
    train_data, eval_data = split_training_data(
        prices=prices,
        ohlcv=ohlcv,
        regimes=regimes,
        micro_indicators=micro,
        macro_indicators=macro,
        eval_ratio=config.eval_ratio,
        window_size=config.window_size,
        split_index=split_index,
    )

    def make_train_env(rank: int = 0):
        def _factory():
            env = Monitor(
                PPOPortfolioEnv(
                    prices=train_data["prices"],
                    ohlcv=train_data["ohlcv"],
                    regimes=train_data["regimes"],
                    micro_indicators=train_data["micro"],
                    macro_indicators=train_data["macro"],
                    window_size=config.window_size,
                    episode_length=min(
                        config.episode_length,
                        train_data["prices"].shape[0] - config.window_size - 2,
                    ),
                    risk_low=config.risk_low,
                    risk_high=config.risk_high,
                    transaction_fee=config.transaction_fee,
                    turnover_penalty=config.turnover_penalty,
                    variance_penalty_scale=config.variance_penalty_scale,
                    concentration_penalty_scale=config.concentration_penalty_scale,
                    drawdown_penalty_scale=config.drawdown_penalty_scale,
                    downside_penalty_scale=config.downside_penalty_scale,
                    benchmark_reward_scale=config.benchmark_reward_scale,
                    horizon_reward_scale=config.horizon_reward_scale,
                    diversification_reward_scale=config.diversification_reward_scale,
                    target_daily_volatility=config.target_daily_volatility,
                    target_volatility_penalty_scale=config.target_volatility_penalty_scale,
                    max_asset_weight=config.max_asset_weight,
                    max_cash_weight=config.max_cash_weight,
                    reward_scale=config.reward_scale,
                    random_start=config.random_start,
                    risky_asset_count=risky_asset_count,
                )
            )
            env.reset(seed=config.seed + rank)
            return env

        return _factory

    def make_eval_env():
        return PPOPortfolioEnv(
            prices=eval_data["prices"],
            ohlcv=eval_data["ohlcv"],
            regimes=eval_data["regimes"],
            micro_indicators=eval_data["micro"],
            macro_indicators=eval_data["macro"],
            window_size=config.window_size,
            episode_length=min(
                config.episode_length,
                eval_data["prices"].shape[0] - config.window_size - 2,
            ),
            risk_low=config.risk_low,
            risk_high=config.risk_high,
            transaction_fee=config.transaction_fee,
            turnover_penalty=config.turnover_penalty,
            variance_penalty_scale=config.variance_penalty_scale,
            concentration_penalty_scale=config.concentration_penalty_scale,
            drawdown_penalty_scale=config.drawdown_penalty_scale,
            downside_penalty_scale=config.downside_penalty_scale,
            benchmark_reward_scale=config.benchmark_reward_scale,
            horizon_reward_scale=config.horizon_reward_scale,
            diversification_reward_scale=config.diversification_reward_scale,
            target_daily_volatility=config.target_daily_volatility,
            target_volatility_penalty_scale=config.target_volatility_penalty_scale,
            max_asset_weight=config.max_asset_weight,
            max_cash_weight=config.max_cash_weight,
            reward_scale=config.reward_scale,
            random_start=False,
            fixed_risk=0.5,
            risky_asset_count=risky_asset_count,
        )

    train_env = DummyVecEnv(
        [make_train_env(index) for index in range(max(int(config.n_envs), 1))]
    )
    eval_env = DummyVecEnv([lambda: Monitor(make_eval_env())])

    with tempfile.TemporaryDirectory(prefix=f"stockify_{asset_class}_ppo_") as tmp_dir:
        callback = EvalCallback(
            eval_env,
            best_model_save_path=tmp_dir,
            log_path=tmp_dir,
            eval_freq=max(config.eval_freq, 1),
            deterministic=True,
            render=False,
        )
        policy_kwargs = {
            "net_arch": {"pi": list(config.policy_layers), "vf": list(config.policy_layers)},
            "activation_fn": _activation_class(config.activation_fn),
            "ortho_init": bool(config.orthogonal_init),
        }
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=_scheduled_value(
                initial=config.learning_rate,
                final=config.learning_rate_final,
                schedule=config.learning_rate_schedule,
            ),
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=_scheduled_value(
                initial=config.clip_range,
                final=config.clip_range_final,
                schedule=config.clip_range_schedule,
            ),
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            normalize_advantage=config.normalize_advantage,
            use_sde=config.use_sde,
            sde_sample_freq=config.sde_sample_freq,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl if config.target_kl and config.target_kl > 0 else None,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=config.seed,
            device=config.device,
        )
        model.learn(total_timesteps=config.total_timesteps, callback=callback, progress_bar=False)

        best_model_path = Path(tmp_dir) / "best_model.zip"
        trained_model = PPO.load(best_model_path, device=config.device) if best_model_path.exists() else model
        evaluation_metrics = _evaluate_model(
            trained_model,
            env_factory=make_eval_env,
            episodes=config.eval_episodes,
        )

    train_env.close()
    eval_env.close()

    model_path = asset_dir / "model.zip"
    backup_path = None
    if model_path.exists():
        backup_path = asset_dir / (
            f"model.backup_before_retrain_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.zip"
        )
        shutil.copy2(model_path, backup_path)

    trained_model.save(model_path)

    metadata = _load_json(asset_dir / "metadata.json")
    feature_names = _load_json(asset_dir / "feature_names.json", {})
    metadata.update(
        {
            "asset_class": asset_class,
            "algorithm": "ppo",
            "policy_backend": "sb3",
            "model_file": "model.zip",
            "feature_version": (
                "ohlcv-stationary-v4-cash-sleeve"
                if config.cash_enabled
                else "ohlcv-stationary-v3"
            ),
            "sub_agent_architecture": {
                "cash_sleeve_enabled": config.cash_enabled,
                "objective": "horizon_return_plus_diversification",
                "risky_asset_count": risky_asset_count,
                "action_dim": int(train_data["prices"].shape[1]),
                "risk_cash_cap": "0.08 + (1 - risk) * 0.42, capped by max_cash_weight",
                "risk_concentration_target": "equal_weight_hhi + risk * (0.45 - equal_weight_hhi)",
            },
            "single_agent_feature_engineering": {
                "market_features": "stationary_returns_volatility_ohlcv_drawdown",
                "indicator_scaler": "RobustScaler(quantile_range=(5, 95)) fit on training split only",
                "feature_clip": config.feature_clip,
                "min_feature_variance": config.min_feature_variance,
            },
            "feature_selection": asdict(feature_report),
            "macro_feature_names": feature_names.get("macro", []),
            "micro_feature_count": int(micro.shape[1]),
            "macro_feature_count": int(macro.shape[1]),
            "ppo_retrained_at": datetime.now(UTC).isoformat(),
            "ppo_training_config": asdict(config),
            "action_dim": int(train_data["prices"].shape[1]),
            **evaluation_metrics,
            "train_rows": int(train_data["prices"].shape[0]),
            "eval_rows": int(eval_data["prices"].shape[0]),
            "tickers": tickers,
        }
    )
    _save_json(asset_dir / "metadata.json", metadata)

    report = PPOTrainingReport(
        asset_class=asset_class,
        total_timesteps=config.total_timesteps,
        train_rows=int(train_data["prices"].shape[0]),
        eval_rows=int(eval_data["prices"].shape[0]),
        eval_mean_reward=evaluation_metrics["eval_mean_reward"],
        eval_mean_final_value=evaluation_metrics["eval_mean_final_value"],
        eval_mean_sharpe=evaluation_metrics["eval_mean_sharpe"],
        eval_mean_max_drawdown=evaluation_metrics["eval_mean_max_drawdown"],
        eval_mean_benchmark_alpha=evaluation_metrics["eval_mean_benchmark_alpha"],
        model_path=str(model_path),
        backup_path=str(backup_path) if backup_path is not None else None,
        trained_at=datetime.now(UTC).isoformat(),
    )
    _save_json(asset_dir / "training_summary.json", asdict(report))
    return report
