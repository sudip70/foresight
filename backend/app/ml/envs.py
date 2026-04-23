from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium import spaces


SINGLE_AGENT_PRICE_FEATURES = (
    "log_return_1d",
    "log_return_5d",
    "log_return_21d",
    "log_return_63d",
    "realized_volatility_21d",
    "realized_volatility_63d",
    "intraday_range",
    "close_to_open_return",
    "log_volume_ratio_21d",
    "drawdown_63d",
    "sma_ratio_21d",
)
SINGLE_AGENT_MARKET_BLOCKS = len(SINGLE_AGENT_PRICE_FEATURES) + 1


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.asarray(numerator, dtype=np.float32) / np.clip(
        np.asarray(denominator, dtype=np.float32),
        1e-12,
        None,
    )


def _log_return(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return np.log(_safe_ratio(current, previous))


def _rolling_volatility(prices: np.ndarray, step: int, window: int) -> np.ndarray:
    start = max(0, step - window + 1)
    window_prices = np.clip(np.asarray(prices[start : step + 1], dtype=np.float32), 1e-12, None)
    if window_prices.shape[0] < 2:
        return np.zeros(window_prices.shape[1], dtype=np.float32)
    returns = np.diff(np.log(window_prices), axis=0)
    return np.std(returns, axis=0).astype(np.float32)


def _rolling_log_return(prices: np.ndarray, step: int, window: int) -> np.ndarray:
    current = prices[step]
    anchor = prices[max(0, step - window)]
    return _log_return(current, anchor)


def _rolling_drawdown(prices: np.ndarray, step: int, window: int) -> np.ndarray:
    start = max(0, step - window + 1)
    window_prices = np.asarray(prices[start : step + 1], dtype=np.float32)
    rolling_high = np.max(window_prices, axis=0)
    return (_safe_ratio(prices[step], rolling_high) - 1.0).astype(np.float32)


def _rolling_mean_ratio(prices: np.ndarray, step: int, window: int) -> np.ndarray:
    start = max(0, step - window + 1)
    rolling_mean = np.mean(np.asarray(prices[start : step + 1], dtype=np.float32), axis=0)
    return (_safe_ratio(prices[step], rolling_mean) - 1.0).astype(np.float32)


def _rolling_volume_ratio(ohlcv: np.ndarray, step: int, window: int) -> np.ndarray:
    if ohlcv is None:
        return np.zeros(1, dtype=np.float32)
    volume = np.asarray(ohlcv[:, :, 4], dtype=np.float32)
    start = max(0, step - window + 1)
    rolling_mean = np.mean(volume[start : step + 1], axis=0)
    return _safe_ratio(volume[step], rolling_mean)


def single_agent_observation_dim(
    *,
    n_assets: int,
    micro_dim: int,
    macro_dim: int,
    num_regimes: int = 3,
) -> int:
    return (
        (int(n_assets) * SINGLE_AGENT_MARKET_BLOCKS)
        + 2
        + int(num_regimes)
        + int(micro_dim)
        + int(macro_dim)
    )


def build_single_agent_observation(
    *,
    prices: np.ndarray,
    regimes: np.ndarray,
    micro_indicators: np.ndarray,
    macro_indicators: np.ndarray,
    ohlcv: np.ndarray | None,
    step: int,
    risk_appetite: float,
    prev_weights: np.ndarray | None = None,
    portfolio_value_ratio: float = 1.0,
    num_regimes: int = 3,
) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float32)
    n_assets = prices.shape[1]
    previous_close = prices[max(0, step - 1)]
    close_5 = prices[max(0, step - 5)]
    current_close = prices[step]
    one_day_return = _log_return(current_close, previous_close)
    five_day_return = _log_return(current_close, close_5)
    twenty_one_day_return = _rolling_log_return(prices, step, window=21)
    sixty_three_day_return = _rolling_log_return(prices, step, window=63)
    rolling_volatility_21 = _rolling_volatility(prices, step, window=21)
    rolling_volatility_63 = _rolling_volatility(prices, step, window=63)
    drawdown_63 = _rolling_drawdown(prices, step, window=63)
    sma_ratio_21 = _rolling_mean_ratio(prices, step, window=21)
    prev_weights = (
        np.asarray(prev_weights, dtype=np.float32)
        if prev_weights is not None
        else np.ones(n_assets, dtype=np.float32) / n_assets
    )

    if ohlcv is not None:
        ohlcv = np.asarray(ohlcv, dtype=np.float32)
        current_open = ohlcv[step, :, 0]
        current_high = ohlcv[step, :, 1]
        current_low = ohlcv[step, :, 2]
        intraday_range = _safe_ratio(current_high - current_low, current_close)
        close_to_open = _safe_ratio(current_close, current_open) - 1.0
        volume_ratio = np.log(np.clip(_rolling_volume_ratio(ohlcv, step, window=21), 1e-6, None))
    else:
        intraday_range = np.zeros(n_assets, dtype=np.float32)
        close_to_open = np.zeros(n_assets, dtype=np.float32)
        volume_ratio = np.zeros(n_assets, dtype=np.float32)

    regime_one_hot = np.zeros(int(num_regimes), dtype=np.float32)
    regime_one_hot[int(regimes[step])] = 1.0
    observation = np.concatenate(
        [
            one_day_return,
            five_day_return,
            twenty_one_day_return,
            sixty_three_day_return,
            rolling_volatility_21,
            rolling_volatility_63,
            intraday_range,
            close_to_open,
            volume_ratio,
            drawdown_63,
            sma_ratio_21,
            prev_weights,
            np.array([portfolio_value_ratio], dtype=np.float32),
            np.array([risk_appetite], dtype=np.float32),
            regime_one_hot,
            np.asarray(micro_indicators[step], dtype=np.float32),
            np.asarray(macro_indicators[step], dtype=np.float32),
        ]
    )
    return observation.astype(np.float32)


def meta_observation_dim(
    *,
    n_assets: int,
    micro_dim: int,
    macro_dim: int,
    class_feature_dim: int = 9,
    num_regimes: int = 3,
) -> int:
    return (
        (int(n_assets) * 4)
        + int(class_feature_dim)
        + 2
        + int(micro_dim)
        + int(macro_dim)
        + int(num_regimes)
    )


def build_meta_observation(
    *,
    step: int,
    mu: np.ndarray,
    cov_diag: np.ndarray,
    prev_weights: np.ndarray,
    sub_agent_weights: np.ndarray,
    class_features: np.ndarray,
    micro_indicators: np.ndarray,
    macro_indicators: np.ndarray,
    regimes: np.ndarray,
    risk_appetite: float,
    portfolio_value_ratio: float = 1.0,
    num_regimes: int = 3,
) -> np.ndarray:
    regime_one_hot = np.zeros(int(num_regimes), dtype=np.float32)
    regime_one_hot[int(regimes[step])] = 1.0
    observation = np.concatenate(
        [
            np.asarray(mu, dtype=np.float32),
            np.asarray(cov_diag, dtype=np.float32),
            np.asarray(prev_weights, dtype=np.float32),
            np.asarray(sub_agent_weights, dtype=np.float32),
            np.asarray(class_features, dtype=np.float32),
            np.array([portfolio_value_ratio], dtype=np.float32),
            np.array([risk_appetite], dtype=np.float32),
            np.asarray(micro_indicators[step], dtype=np.float32),
            np.asarray(macro_indicators[step], dtype=np.float32),
            regime_one_hot,
        ]
    )
    return observation.astype(np.float32)


class SingleAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        prices: np.ndarray,
        regimes: np.ndarray,
        micro_indicators: np.ndarray,
        macro_indicators: np.ndarray,
        ohlcv: np.ndarray | None = None,
        risk_appetite: float,
    ) -> None:
        super().__init__()
        self.prices = np.asarray(prices, dtype=float)
        self.regimes = np.asarray(regimes, dtype=int)
        self.micro_indicators = np.asarray(micro_indicators, dtype=float)
        self.macro_indicators = np.asarray(macro_indicators, dtype=float)
        self.ohlcv = None if ohlcv is None else np.asarray(ohlcv, dtype=float)
        self.risk_appetite = float(risk_appetite)
        self.num_regimes = 3
        self.n_assets = self.prices.shape[1]
        obs_dim = single_agent_observation_dim(
            n_assets=self.n_assets,
            micro_dim=self.micro_indicators.shape[1],
            macro_dim=self.macro_indicators.shape[1],
            num_regimes=self.num_regimes,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def observation_at(
        self,
        step: int,
        *,
        prev_weights: np.ndarray | None = None,
        portfolio_value_ratio: float = 1.0,
    ) -> np.ndarray:
        return build_single_agent_observation(
            prices=self.prices,
            regimes=self.regimes,
            micro_indicators=self.micro_indicators,
            macro_indicators=self.macro_indicators,
            ohlcv=self.ohlcv,
            step=step,
            risk_appetite=self.risk_appetite,
            prev_weights=prev_weights,
            portfolio_value_ratio=portfolio_value_ratio,
            num_regimes=self.num_regimes,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return self.observation_at(0), {}

    def step(self, action):  # pragma: no cover - not used in phase-1 runtime
        return self.observation_at(-1), 0.0, True, False, {}


class MetaPortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        micro_indicators: np.ndarray,
        macro_indicators: np.ndarray,
        regimes: np.ndarray,
        n_assets: int,
        class_feature_dim: int = 9,
    ) -> None:
        super().__init__()
        self.micro_indicators = np.asarray(micro_indicators, dtype=float)
        self.macro_indicators = np.asarray(macro_indicators, dtype=float)
        self.regimes = np.asarray(regimes, dtype=int)
        self.n_assets = int(n_assets)
        self.class_feature_dim = int(class_feature_dim)
        self.num_regimes = 3
        obs_dim = meta_observation_dim(
            n_assets=self.n_assets,
            micro_dim=self.micro_indicators.shape[1],
            macro_dim=self.macro_indicators.shape[1],
            class_feature_dim=self.class_feature_dim,
            num_regimes=self.num_regimes,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def observation_at(
        self,
        step: int,
        *,
        mu: np.ndarray,
        cov_diag: np.ndarray,
        prev_weights: np.ndarray,
        sub_agent_weights: np.ndarray,
        class_features: np.ndarray,
        risk_appetite: float,
        portfolio_value_ratio: float = 1.0,
    ) -> np.ndarray:
        return build_meta_observation(
            step=step,
            mu=mu,
            cov_diag=cov_diag,
            prev_weights=prev_weights,
            sub_agent_weights=sub_agent_weights,
            class_features=class_features,
            micro_indicators=self.micro_indicators,
            macro_indicators=self.macro_indicators,
            regimes=self.regimes,
            risk_appetite=risk_appetite,
            portfolio_value_ratio=portfolio_value_ratio,
            num_regimes=self.num_regimes,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):  # pragma: no cover - not used directly in phase 1
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}
