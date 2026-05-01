from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
import os
import shutil
import tempfile

import gymnasium as gym
from gymnasium import spaces
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.app.ml.artifacts import ASSET_CLASSES, AssetArtifacts, load_asset_artifacts
from backend.app.ml.envs import (
    SingleAgentEnv,
    build_meta_observation,
    meta_observation_dim,
)
from backend.app.ml.policies import (
    apply_class_guardrails,
    normalize_action_with_cash_sleeve,
    normalize_weights,
    normalize_weights_with_caps,
)


CASH_ASSET_CLASS = "cash"
META_V3_FEATURE_VERSION = "sac-meta-v3-globalmacro-cashaware"
META_CLASS_FEATURE_DIM = 12
META_SLEEVE_ACTION_LAYOUT = "stock_crypto_etf_cash_sleeves"


@dataclass(frozen=True)
class SACMetaTrainingConfig:
    total_timesteps: int = 150_000
    eval_freq: int = 5_000
    episode_length: int = 252
    window_size: int = 60
    eval_ratio: float = 0.2
    risk_values: tuple[float, ...] = (0.3, 0.5, 0.7)
    transaction_fee: float = 0.001
    turnover_penalty: float = 0.001
    volatility_penalty_scale: float = 0.5
    concentration_penalty_scale: float = 0.01
    drawdown_penalty_scale: float = 0.05
    downside_penalty_scale: float = 0.25
    benchmark_reward_scale: float = 0.5
    horizon_reward_scale: float = 1.0
    diversification_reward_scale: float = 0.10
    target_daily_volatility: float | None = 0.014
    target_volatility_penalty_scale: float = 0.25
    reward_scale: float = 100.0
    max_asset_weight: float = 0.20
    max_stock_weight: float = 0.85
    max_crypto_weight: float = 0.30
    max_etf_weight: float = 0.70
    max_cash_weight: float = 0.95
    min_expected_daily_return: float = 0.0
    cash_shortfall_penalty_scale: float = 0.0
    cash_enabled: bool = True
    cash_annual_return: float = 0.04
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    learning_starts: int = 2_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str | float = "auto"
    policy_layers: tuple[int, ...] = (256, 256)
    activation_fn: str = "relu"
    use_sde: bool = False
    seed: int = 42
    device: str = "cpu"


@dataclass(frozen=True)
class SACMetaTrainingReport:
    total_timesteps: int
    train_rows: int
    eval_rows: int
    eval_mean_reward: float
    eval_mean_final_value: float
    eval_mean_sharpe: float
    eval_mean_max_drawdown: float
    eval_mean_agent_alpha: float
    model_path: str
    backup_path: str | None
    trained_at: str


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _stage_artifact_dir(asset_dir: Path) -> Path:
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{asset_dir.name}.staging.",
            dir=str(asset_dir.parent),
        )
    )
    shutil.copytree(asset_dir, staging_dir, dirs_exist_ok=True)
    return staging_dir


def _commit_staged_artifact_dir(staging_dir: Path, asset_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = asset_dir.with_name(f".{asset_dir.name}.backup_before_commit_{timestamp}")
    os.rename(asset_dir, backup_dir)
    try:
        os.rename(staging_dir, asset_dir)
    except Exception:
        os.rename(backup_dir, asset_dir)
        raise
    return backup_dir


def _common_date_positions(dates: np.ndarray, common_dates: np.ndarray) -> np.ndarray:
    order = np.argsort(dates)
    sorted_dates = dates[order]
    positions = np.searchsorted(sorted_dates, common_dates)
    if (
        np.any(positions >= sorted_dates.shape[0])
        or not np.array_equal(sorted_dates[positions], common_dates)
    ):
        raise ValueError("Unable to align artifacts by common dates")
    return order[positions]


def _class_ranges(assets: dict[str, AssetArtifacts]) -> dict[str, tuple[int, int]]:
    cursor = 0
    ranges: dict[str, tuple[int, int]] = {}
    for asset_class in ASSET_CLASSES:
        width = len(assets[asset_class].tickers)
        ranges[asset_class] = (cursor, cursor + width)
        cursor += width
    ranges[CASH_ASSET_CLASS] = (cursor, cursor + 1)
    return ranges


def _cash_daily_return(config: SACMetaTrainingConfig) -> float:
    return (1.0 + config.cash_annual_return) ** (1.0 / 252.0) - 1.0


def _risk_cash_cap(risk: float, *, configured_max: float) -> float:
    risk_value = float(np.clip(risk, 0.0, 1.0))
    dynamic_cap = 0.08 + ((1.0 - risk_value) * 0.42)
    return float(min(float(configured_max), dynamic_cap))


def _risk_class_caps(config: SACMetaTrainingConfig, risk: float) -> dict[str, float]:
    risk_value = float(np.clip(risk, 0.0, 1.0))
    caps = {
        "stock": min(config.max_stock_weight, 0.55 + (0.25 * risk_value)),
        "crypto": min(config.max_crypto_weight, 0.04 + (0.31 * risk_value)),
        "etf": min(config.max_etf_weight, 0.72),
    }
    if config.cash_enabled:
        caps[CASH_ASSET_CLASS] = _risk_cash_cap(risk_value, configured_max=config.max_cash_weight)
    return caps


def _risk_class_floors(config: SACMetaTrainingConfig, risk: float) -> dict[str, float]:
    risk_value = float(np.clip(risk, 0.0, 1.0))
    floors = {
        "stock": 0.12 + (0.08 * risk_value),
        "crypto": 0.02 * risk_value,
        "etf": 0.12 + (0.20 * (1.0 - risk_value)),
    }
    if config.cash_enabled:
        floors[CASH_ASSET_CLASS] = 0.03 * (1.0 - risk_value)
    return floors


def _risk_concentration_target(n_assets: int, risk: float) -> float:
    risk_value = float(np.clip(risk, 0.0, 1.0))
    equal_concentration = 1.0 / max(int(n_assets), 1)
    return equal_concentration + (risk_value * (0.45 - equal_concentration))


def _sub_agent_max_weight(bundle: AssetArtifacts) -> float | None:
    config = bundle.metadata.get("ppo_training_config", {})
    max_weight = config.get("max_asset_weight")
    return None if max_weight is None else float(max_weight)


def _sub_agent_max_cash_weight(bundle: AssetArtifacts) -> float | None:
    config = bundle.metadata.get("ppo_training_config", {})
    max_weight = config.get("max_cash_weight")
    return None if max_weight is None else float(max_weight)


def _compute_mu_cov_diag(prices: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    rows, n_assets = prices.shape
    mu = np.zeros((rows, n_assets), dtype=float)
    cov_diag = np.zeros((rows, n_assets), dtype=float)
    for step in range(rows):
        start = max(0, step - window_size)
        window_prices = prices[start : step + 1]
        if window_prices.shape[0] < 2:
            continue
        returns = np.diff(np.log(np.clip(window_prices, 1e-12, None)), axis=0)
        mu[step] = np.mean(returns, axis=0)
        cov_diag[step] = np.var(returns, axis=0) + 1e-6
    return mu, cov_diag


def _transform_macro_block(
    matrix: np.ndarray,
    scaler: RobustScaler,
) -> np.ndarray:
    transformed = scaler.transform(np.asarray(matrix, dtype=float))
    transformed = np.clip(transformed, -5.0, 5.0)
    return np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(float)


def _build_context(artifact_root: Path, config: SACMetaTrainingConfig) -> dict:
    assets = {
        asset_class: load_asset_artifacts(artifact_root, asset_class, strict=True)
        for asset_class in ASSET_CLASSES
    }
    date_arrays = [np.asarray(bundle.dates) for bundle in assets.values()]
    common_dates = date_arrays[0]
    for dates in date_arrays[1:]:
        common_dates = np.intersect1d(common_dates, dates)

    aligned_assets = {}
    for asset_class, bundle in assets.items():
        positions = _common_date_positions(bundle.dates, common_dates)
        aligned_assets[asset_class] = {
            "prices": bundle.prices[positions],
            "risky_prices": bundle.prices[positions, : len(bundle.tickers)],
            "ohlcv": bundle.ohlcv[positions],
            "regimes": bundle.regimes[positions],
            "micro": bundle.micro_indicators[positions],
            "macro": bundle.macro_indicators[positions],
            "macro_raw": bundle.macro_indicators_raw[positions],
        }

    risky_prices = np.hstack([aligned_assets[name]["risky_prices"] for name in ASSET_CLASSES])
    if config.cash_enabled:
        cash_prices = np.cumprod(
            np.full(common_dates.shape[0], 1.0 + _cash_daily_return(config), dtype=float)
        ).reshape(-1, 1)
        prices = np.hstack([risky_prices, cash_prices])
    else:
        prices = risky_prices

    class_ranges = _class_ranges(assets)
    if not config.cash_enabled:
        class_ranges.pop(CASH_ASSET_CLASS, None)

    global_macro_source = max(
        ASSET_CLASSES,
        key=lambda name: aligned_assets[name]["macro_raw"].shape[1],
    )
    return {
        "assets": assets,
        "aligned_assets": aligned_assets,
        "dates": common_dates,
        "prices": prices,
        "micro": np.hstack([aligned_assets[name]["micro"] for name in ASSET_CLASSES]),
        "macro": np.asarray(aligned_assets[global_macro_source]["macro_raw"], dtype=float),
        "regimes": _mode_regimes(
            np.vstack([aligned_assets[name]["regimes"] for name in ASSET_CLASSES])
        ),
        "class_ranges": class_ranges,
        "global_macro_feature_names": assets[global_macro_source].metadata.get(
            "raw_macro_feature_names",
            assets[global_macro_source].metadata.get("macro_feature_names", []),
        ),
    }


def _mode_regimes(regime_stack: np.ndarray) -> np.ndarray:
    result = []
    for column in regime_stack.T:
        counts = np.bincount(column.astype(int), minlength=3)
        result.append(int(np.argmax(counts)))
    return np.asarray(result, dtype=int)


def _slice_context(context: dict, data_slice: slice) -> dict:
    aligned_assets = {
        asset_class: {
            key: value[data_slice]
            for key, value in asset_context.items()
        }
        for asset_class, asset_context in context["aligned_assets"].items()
    }
    return {
        **context,
        "aligned_assets": aligned_assets,
        "dates": context["dates"][data_slice],
        "prices": context["prices"][data_slice],
        "micro": context["micro"][data_slice],
        "macro": context["macro"][data_slice],
        "regimes": context["regimes"][data_slice],
        "global_macro_feature_names": context["global_macro_feature_names"],
    }


def split_meta_context(context: dict, config: SACMetaTrainingConfig) -> tuple[dict, dict]:
    rows = context["prices"].shape[0]
    split_index = int(rows * (1.0 - config.eval_ratio))
    split_index = max(split_index, config.window_size + 64)
    split_index = min(split_index, rows - (config.window_size + 64))
    if split_index <= config.window_size or split_index >= rows:
        raise ValueError(f"Unable to split meta rows={rows}")
    return (
        _slice_context(context, slice(0, split_index)),
        _slice_context(context, slice(split_index - config.window_size, rows)),
    )


def _fit_meta_macro_scaler(train_context: dict, eval_context: dict) -> tuple[RobustScaler, dict, dict]:
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    scaler.fit(np.asarray(train_context["macro"], dtype=float))
    train_scaled = _transform_macro_block(train_context["macro"], scaler)
    eval_scaled = _transform_macro_block(eval_context["macro"], scaler)
    return (
        scaler,
        {**train_context, "macro": train_scaled},
        {**eval_context, "macro": eval_scaled},
    )


def _predict_sub_agent_allocation(
    *,
    bundle: AssetArtifacts,
    env: SingleAgentEnv,
    prev_weights: np.ndarray | None,
    step: int,
) -> dict:
    observation = env.observation_at(step, prev_weights=prev_weights)
    full_weights = normalize_action_with_cash_sleeve(
        bundle.policy.predict(observation),
        risky_asset_count=len(bundle.tickers),
        max_risky_weight=_sub_agent_max_weight(bundle),
        max_cash_weight=_sub_agent_max_cash_weight(bundle),
    )
    risky_weights = np.asarray(full_weights[: len(bundle.tickers)], dtype=float)
    return {
        "full_weights": np.asarray(full_weights, dtype=float),
        "weights": risky_weights,
        "cash_weight": max(1.0 - float(risky_weights.sum()), 0.0),
    }


def _compose_meta_sub_agent_signal(
    *,
    asset_outputs: dict[str, dict],
    cash_enabled: bool,
) -> np.ndarray:
    class_count = float(len(ASSET_CLASSES))
    risky_signal = np.concatenate(
        [asset_outputs[asset_class]["weights"] / class_count for asset_class in ASSET_CLASSES]
    ).astype(float)
    if not cash_enabled:
        return normalize_weights(risky_signal)

    cash_prior = float(
        np.mean([asset_outputs[asset_class]["cash_weight"] for asset_class in ASSET_CLASSES])
    )
    combined = np.concatenate([risky_signal, np.array([cash_prior], dtype=float)])
    total = float(combined.sum())
    if total > 1e-12:
        combined = combined / total
    return combined


def _precompute_sub_agent_outputs(context: dict, config: SACMetaTrainingConfig) -> dict[float, dict]:
    outputs: dict[float, dict] = {}
    rows = context["prices"].shape[0]
    for risk in config.risk_values:
        previous_by_class: dict[str, np.ndarray] | None = None
        meta_signal = np.zeros_like(context["prices"], dtype=float)
        risky_weights = {
            asset_class: np.zeros((rows, len(context["assets"][asset_class].tickers)), dtype=float)
            for asset_class in ASSET_CLASSES
        }
        cash_weights = {
            asset_class: np.zeros(rows, dtype=float)
            for asset_class in ASSET_CLASSES
        }
        envs = {
            asset_class: SingleAgentEnv(
                prices=context["aligned_assets"][asset_class]["prices"],
                ohlcv=context["aligned_assets"][asset_class]["ohlcv"],
                regimes=context["aligned_assets"][asset_class]["regimes"],
                micro_indicators=context["aligned_assets"][asset_class]["micro"],
                macro_indicators=context["aligned_assets"][asset_class]["macro"],
                risk_appetite=float(risk),
            )
            for asset_class in ASSET_CLASSES
        }
        for step in range(rows):
            asset_outputs = {}
            next_previous: dict[str, np.ndarray] = {}
            for asset_class in ASSET_CLASSES:
                bundle = context["assets"][asset_class]
                previous = previous_by_class.get(asset_class) if previous_by_class else None
                output = _predict_sub_agent_allocation(
                    bundle=bundle,
                    env=envs[asset_class],
                    prev_weights=previous,
                    step=step,
                )
                asset_outputs[asset_class] = output
                next_previous[asset_class] = output["full_weights"]
                risky_weights[asset_class][step] = output["weights"]
                cash_weights[asset_class][step] = output["cash_weight"]
            previous_by_class = next_previous
            meta_signal[step] = _compose_meta_sub_agent_signal(
                asset_outputs=asset_outputs,
                cash_enabled=config.cash_enabled,
            )
        outputs[float(risk)] = {
            "meta_signal": meta_signal,
            "risky_weights": risky_weights,
            "cash_weights": cash_weights,
        }
    return outputs


def _activation_class(name: str):
    from torch import nn

    lookup = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    key = name.lower().replace("-", "_")
    if key not in lookup:
        raise ValueError(f"Unsupported activation_fn: {name}")
    return lookup[key]


class SACMetaPortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        context: dict,
        config: SACMetaTrainingConfig,
        random_start: bool,
        fixed_risk: float | None = None,
    ) -> None:
        super().__init__()
        self.context = context
        self.config = config
        self.random_start = bool(random_start)
        self.fixed_risk = fixed_risk
        self.prices = np.asarray(context["prices"], dtype=float)
        self.micro = np.asarray(context["micro"], dtype=float)
        self.macro = np.asarray(context["macro"], dtype=float)
        self.regimes = np.asarray(context["regimes"], dtype=int)
        self.class_ranges = context["class_ranges"]
        self.sleeve_names = list(ASSET_CLASSES)
        if config.cash_enabled:
            self.sleeve_names.append(CASH_ASSET_CLASS)
        self.mu, self.cov_diag = _compute_mu_cov_diag(self.prices, config.window_size)
        self.sub_agent_outputs_by_risk = _precompute_sub_agent_outputs(context, config)
        self.risk_values = tuple(float(value) for value in config.risk_values)
        self.n_assets = int(self.prices.shape[1])
        self.max_start = self.prices.shape[0] - config.episode_length - 2
        if self.max_start < config.window_size:
            raise ValueError("Not enough rows to train SAC meta env")

        obs_dim = meta_observation_dim(
            n_assets=self.n_assets,
            micro_dim=self.micro.shape[1],
            macro_dim=self.macro.shape[1],
            class_feature_dim=META_CLASS_FEATURE_DIM,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.sleeve_names),),
            dtype=np.float32,
        )
        self.current_step = config.window_size
        self.end_step = self.current_step + config.episode_length
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.prev_weights = np.ones(self.n_assets, dtype=float) / self.n_assets
        self.risk_appetite = 0.5
        self.episode_log_return = 0.0
        self.episode_agent_log_return = 0.0
        self.concentration_history: list[float] = []

    def _sample_risk(self) -> float:
        if self.fixed_risk is not None:
            return float(self.fixed_risk)
        index = int(self.np_random.integers(0, len(self.risk_values)))
        return self.risk_values[index]

    def _risk_key(self) -> float:
        return min(self.risk_values, key=lambda value: abs(value - self.risk_appetite))

    def _sub_agent_snapshot(self) -> dict:
        return self.sub_agent_outputs_by_risk[self._risk_key()]

    def _sub_agent_weights(self) -> np.ndarray:
        return self._sub_agent_snapshot()["meta_signal"][self.current_step]

    def _class_features(self) -> np.ndarray:
        snapshot = self._sub_agent_snapshot()
        features = []
        for asset_class in ASSET_CLASSES:
            start, end = self.class_ranges[asset_class]
            class_signal = snapshot["risky_weights"][asset_class][self.current_step]
            class_weights = normalize_weights(class_signal)
            class_mu = self.mu[self.current_step, start:end]
            class_cov_diag = self.cov_diag[self.current_step, start:end]
            class_expected_return = float(np.dot(class_weights, class_mu))
            class_volatility = float(np.sqrt(max(np.dot(class_weights**2, class_cov_diag), 0.0)))
            class_prev_weight = float(self.prev_weights[start:end].sum())
            class_cash_weight = float(snapshot["cash_weights"][asset_class][self.current_step])
            features.extend(
                [class_expected_return, class_volatility, class_prev_weight, class_cash_weight]
            )
        return np.asarray(features, dtype=np.float32)

    def _cash_index(self) -> int | None:
        if not self.config.cash_enabled or CASH_ASSET_CLASS not in self.class_ranges:
            return None
        return self.class_ranges[CASH_ASSET_CLASS][0]

    def _asset_weight_caps(self) -> np.ndarray:
        caps = np.full(self.n_assets, self.config.max_asset_weight, dtype=float)
        cash_index = self._cash_index()
        if cash_index is not None:
            caps[cash_index] = _risk_cash_cap(
                self.risk_appetite,
                configured_max=self.config.max_cash_weight,
            )
        return caps

    def _apply_sleeve_constraints(self, action: np.ndarray) -> dict[str, float]:
        floors = _risk_class_floors(self.config, self.risk_appetite)
        caps = _risk_class_caps(self.config, self.risk_appetite)
        floor_total = min(sum(floors.get(name, 0.0) for name in self.sleeve_names), 0.85)
        remaining = max(1.0 - floor_total, 0.0)
        capacity = np.asarray(
            [
                max(caps.get(name, 1.0) - floors.get(name, 0.0), 0.0)
                for name in self.sleeve_names
            ],
            dtype=float,
        )
        if remaining <= 1e-12:
            constrained = {name: floors.get(name, 0.0) for name in self.sleeve_names}
            total = sum(constrained.values())
            return {name: value / total for name, value in constrained.items()}

        scores = np.clip(np.asarray(action, dtype=float).reshape(-1), 0.0, None)
        if scores.shape[0] != len(self.sleeve_names):
            raise ValueError("SAC meta action dimension must match sleeve count")
        if float(scores.sum()) <= 1e-12:
            scores = np.ones_like(scores)
        caps_for_remaining = capacity / remaining
        if float(caps_for_remaining.sum()) < 1.0:
            caps_for_remaining = caps_for_remaining / max(float(caps_for_remaining.sum()), 1e-12)
        variable = normalize_weights_with_caps(scores, caps_for_remaining)
        constrained = {
            name: floors.get(name, 0.0) + (remaining * float(variable[index]))
            for index, name in enumerate(self.sleeve_names)
        }
        total = sum(constrained.values())
        return {name: value / total for name, value in constrained.items()}

    def _sleeves_to_asset_weights(self, sleeve_weights: dict[str, float]) -> np.ndarray:
        snapshot = self._sub_agent_snapshot()
        weights = np.zeros(self.n_assets, dtype=float)
        for asset_class in ASSET_CLASSES:
            start, end = self.class_ranges[asset_class]
            class_signal = snapshot["risky_weights"][asset_class][self.current_step]
            internal_weights = normalize_weights(class_signal)
            weights[start:end] = sleeve_weights.get(asset_class, 0.0) * internal_weights
        cash_index = self._cash_index()
        if cash_index is not None:
            weights[cash_index] = sleeve_weights.get(CASH_ASSET_CLASS, 0.0)
        return normalize_weights(weights)

    def _apply_action_controls(self, action: np.ndarray) -> tuple[np.ndarray, dict]:
        sleeve_weights = self._apply_sleeve_constraints(action)
        weights = self._sleeves_to_asset_weights(sleeve_weights)
        caps = self._asset_weight_caps()
        for _ in range(3):
            weights = apply_class_guardrails(
                weights,
                class_ranges=self.class_ranges,
                max_class_weights=_risk_class_caps(self.config, self.risk_appetite),
                max_asset_weight=None,
            )
            weights = normalize_weights_with_caps(weights, caps)
        return weights, {"sleeve_weights": sleeve_weights}

    def _observation(self) -> np.ndarray:
        sub_agent_weights = self._sub_agent_weights()
        return build_meta_observation(
            step=self.current_step,
            mu=self.mu[self.current_step],
            cov_diag=self.cov_diag[self.current_step],
            prev_weights=self.prev_weights,
            sub_agent_weights=sub_agent_weights,
            class_features=self._class_features(),
            micro_indicators=self.micro,
            macro_indicators=self.macro,
            regimes=self.regimes,
            risk_appetite=self.risk_appetite,
            portfolio_value_ratio=self.portfolio_value,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.risk_appetite = self._sample_risk()
        if self.random_start:
            self.current_step = int(
                self.np_random.integers(self.config.window_size, self.max_start + 1)
            )
        else:
            self.current_step = self.config.window_size
        self.end_step = min(
            self.current_step + self.config.episode_length,
            self.prices.shape[0] - 2,
        )
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.prev_weights = np.asarray(self._sub_agent_weights(), dtype=float)
        self.episode_log_return = 0.0
        self.episode_agent_log_return = 0.0
        self.concentration_history = []
        return self._observation(), {}

    def step(self, action):
        sub_agent_weights = self._sub_agent_weights()
        weights, risk_adjustment = self._apply_action_controls(action)
        turnover = float(np.sum(np.abs(weights - self.prev_weights)))
        transaction_cost = turnover * self.config.transaction_fee
        asset_returns = (
            self.prices[self.current_step + 1] - self.prices[self.current_step]
        ) / np.clip(self.prices[self.current_step], 1e-12, None)
        portfolio_return = float(np.dot(weights, asset_returns))
        agent_benchmark_return = float(np.dot(sub_agent_weights, asset_returns))
        portfolio_volatility = float(
            np.sqrt(max(np.dot(weights**2, self.cov_diag[self.current_step]), 0.0))
        )
        concentration = float(np.sum(weights**2))

        previous_drawdown = 1.0 - (self.portfolio_value / max(self.peak_value, 1e-12))
        net_return = max(1.0 + portfolio_return - transaction_cost, 1e-6)
        next_portfolio_value = self.portfolio_value * net_return
        next_peak = max(self.peak_value, next_portfolio_value)
        drawdown = 1.0 - (next_portfolio_value / max(next_peak, 1e-12))
        drawdown_worsening = max(drawdown - previous_drawdown, 0.0)
        downside_return = max(-portfolio_return, 0.0)
        expected_daily_return = float(np.dot(weights, self.mu[self.current_step]))
        expected_return_shortfall = max(
            self.config.min_expected_daily_return - expected_daily_return,
            0.0,
        )
        target_volatility_penalty = 0.0
        if (
            self.config.target_daily_volatility is not None
            and self.config.target_volatility_penalty_scale > 0
        ):
            target_volatility_penalty = self.config.target_volatility_penalty_scale * abs(
                portfolio_volatility - self.config.target_daily_volatility
            )
        self.episode_log_return += float(np.log(net_return))
        self.episode_agent_log_return += float(np.log(max(1.0 + agent_benchmark_return, 1e-6)))
        self.concentration_history.append(concentration)
        terminated = self.current_step + 1 >= self.end_step
        concentration_target = _risk_concentration_target(self.n_assets, self.risk_appetite)
        terminal_reward = 0.0
        if terminated:
            horizon_alpha = self.episode_log_return - self.episode_agent_log_return
            average_concentration = (
                float(np.mean(self.concentration_history))
                if self.concentration_history
                else concentration
            )
            terminal_reward = (
                self.config.horizon_reward_scale * self.episode_log_return
                + self.config.benchmark_reward_scale * horizon_alpha
                - self.config.diversification_reward_scale
                * max(average_concentration - concentration_target, 0.0)
            )

        raw_reward = (
            float(np.log(net_return))
            + self.config.benchmark_reward_scale * (portfolio_return - agent_benchmark_return)
            + terminal_reward
            - self.config.volatility_penalty_scale * portfolio_volatility
            - self.config.turnover_penalty * turnover
            - self.config.concentration_penalty_scale * concentration
            - self.config.downside_penalty_scale * downside_return
            - self.config.drawdown_penalty_scale * (drawdown + drawdown_worsening)
            - target_volatility_penalty
            - self.config.cash_shortfall_penalty_scale * expected_return_shortfall
        )
        reward = self.config.reward_scale * raw_reward

        self.portfolio_value = next_portfolio_value
        self.peak_value = next_peak
        self.prev_weights = weights
        self.current_step += 1
        return self._observation(), float(reward), terminated, False, {
            "portfolio_value": float(self.portfolio_value),
            "portfolio_return": portfolio_return,
            "agent_benchmark_return": agent_benchmark_return,
            "agent_alpha": portfolio_return - agent_benchmark_return,
            "portfolio_volatility": portfolio_volatility,
            "turnover": turnover,
            "drawdown": drawdown,
            "expected_daily_return": expected_daily_return,
            "expected_return_shortfall": expected_return_shortfall,
            "terminal_reward": terminal_reward,
            "horizon_log_return": self.episode_log_return,
            "horizon_agent_log_return": self.episode_agent_log_return,
            "cash_overlay_applied": False,
            "sleeve_weights": risk_adjustment["sleeve_weights"],
            "risk_appetite": self.risk_appetite,
            "crypto_weight": float(
                weights[self.class_ranges["crypto"][0] : self.class_ranges["crypto"][1]].sum()
            ),
        }


def _evaluate_model(model: SAC, *, env_factory, episodes: int = 3) -> dict[str, float]:
    rewards = []
    final_values = []
    sharpes = []
    max_drawdowns = []
    agent_alphas = []
    for _ in range(episodes):
        env = env_factory()
        observation, _ = env.reset()
        done = False
        truncated = False
        reward_sum = 0.0
        returns = []
        drawdowns = []
        alphas = []
        info = {"portfolio_value": 1.0}
        while not done and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, info = env.step(action)
            reward_sum += float(reward)
            returns.append(float(info["portfolio_return"]))
            drawdowns.append(float(info["drawdown"]))
            alphas.append(float(info["agent_alpha"]))
        volatility = float(np.std(returns))
        sharpes.append(
            float((np.mean(returns) / volatility) * np.sqrt(252))
            if volatility > 0
            else 0.0
        )
        rewards.append(reward_sum)
        final_values.append(float(info["portfolio_value"]))
        max_drawdowns.append(float(max(drawdowns) if drawdowns else 0.0))
        agent_alphas.append(float(np.mean(alphas) if alphas else 0.0))
    return {
        "eval_mean_reward": float(np.mean(rewards)),
        "eval_mean_final_value": float(np.mean(final_values)),
        "eval_mean_sharpe": float(np.mean(sharpes)),
        "eval_mean_max_drawdown": float(np.mean(max_drawdowns)),
        "eval_mean_agent_alpha": float(np.mean(agent_alphas)),
    }


def train_meta_agent(
    *,
    artifact_root: Path,
    config: SACMetaTrainingConfig,
) -> SACMetaTrainingReport:
    context = _build_context(artifact_root, config)
    train_context, eval_context = split_meta_context(context, config)
    macro_scaler, train_context, eval_context = _fit_meta_macro_scaler(train_context, eval_context)

    def make_train_env():
        return Monitor(
            SACMetaPortfolioEnv(
                context=train_context,
                config=config,
                random_start=True,
            )
        )

    def make_eval_env():
        return SACMetaPortfolioEnv(
            context=eval_context,
            config=config,
            random_start=False,
            fixed_risk=0.5,
        )

    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([lambda: Monitor(make_eval_env())])

    with tempfile.TemporaryDirectory(prefix="foresight_sac_meta_") as tmp_dir:
        callback = EvalCallback(
            eval_env,
            best_model_save_path=tmp_dir,
            log_path=tmp_dir,
            eval_freq=max(config.eval_freq, 1),
            deterministic=True,
            render=False,
        )
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            tau=config.tau,
            gamma=config.gamma,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            ent_coef=config.ent_coef,
            use_sde=config.use_sde,
            policy_kwargs={
                "net_arch": list(config.policy_layers),
                "activation_fn": _activation_class(config.activation_fn),
            },
            verbose=0,
            seed=config.seed,
            device=config.device,
        )
        model.learn(total_timesteps=config.total_timesteps, callback=callback, progress_bar=False)
        best_model_path = Path(tmp_dir) / "best_model.zip"
        trained_model = SAC.load(best_model_path, device=config.device) if best_model_path.exists() else model
        evaluation = _evaluate_model(trained_model, env_factory=make_eval_env, episodes=3)

    train_env.close()
    eval_env.close()

    meta_dir = artifact_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    staging_meta_dir = _stage_artifact_dir(meta_dir)
    final_model_path = meta_dir / "model.zip"
    staging_model_path = staging_meta_dir / "model.zip"
    backup_path = None
    if final_model_path.exists():
        backup_name = (
            f"model.backup_before_sac_retrain_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.zip"
        )
        backup_path = meta_dir / backup_name
        shutil.copy2(final_model_path, staging_meta_dir / backup_name)
    trained_model.save(staging_model_path)
    joblib.dump(macro_scaler, staging_meta_dir / "meta_macro_scaler.pkl")

    obs_dim = meta_observation_dim(
        n_assets=context["prices"].shape[1],
        micro_dim=context["micro"].shape[1],
        macro_dim=context["macro"].shape[1],
        class_feature_dim=META_CLASS_FEATURE_DIM,
    )
    metadata = {
        "algorithm": "sac",
        "policy_backend": "sb3",
        "model_file": "model.zip",
        "feature_version": META_V3_FEATURE_VERSION,
        "policy_observation_dim": int(obs_dim),
        "inference_observation_dim": int(obs_dim),
        "action_dim": len(ASSET_CLASSES) + (1 if config.cash_enabled else 0),
        "policy_action_layout": META_SLEEVE_ACTION_LAYOUT,
        "policy_action_names": list(ASSET_CLASSES)
        + ([CASH_ASSET_CLASS] if config.cash_enabled else []),
        "class_feature_dim": META_CLASS_FEATURE_DIM,
        "uses_shared_macro": True,
        "meta_macro_feature_names": list(context["global_macro_feature_names"]),
        "class_order": list(ASSET_CLASSES),
        "class_ranges": {
            key: [int(value[0]), int(value[1])]
            for key, value in context["class_ranges"].items()
        },
        "guardrails": {
            "max_asset_weight": config.max_asset_weight,
            "max_stock_weight": config.max_stock_weight,
            "max_crypto_weight": config.max_crypto_weight,
            "max_etf_weight": config.max_etf_weight,
            "max_cash_weight": config.max_cash_weight,
            "min_expected_daily_return": config.min_expected_daily_return,
            "cash_shortfall_penalty_scale": config.cash_shortfall_penalty_scale,
            "cash_enabled": config.cash_enabled,
            "cash_annual_return": config.cash_annual_return,
            "hard_min_stock_weight_removed": True,
        },
        "meta_architecture": {
            "sub_agent_cash_aware": True,
            "shared_global_macro": True,
            "meta_action_layout": META_SLEEVE_ACTION_LAYOUT,
            "class_feature_layout": [
                "class_expected_return",
                "class_volatility",
                "class_previous_weight",
                "class_cash_weight",
            ],
            "sub_agent_signal_layout": "per-class risky weights scaled by class count plus mean cash prior",
        },
        "sac_training_config": asdict(config),
        "sac_retrained_at": datetime.now(UTC).isoformat(),
        "train_rows": int(train_context["prices"].shape[0]),
        "eval_rows": int(eval_context["prices"].shape[0]),
        "train_date_range": {
            "start": str(train_context["dates"][0]),
            "end": str(train_context["dates"][-1]),
        },
        "eval_date_range": {
            "start": str(eval_context["dates"][0]),
            "end": str(eval_context["dates"][-1]),
        },
        **evaluation,
    }
    _save_json(staging_meta_dir / "metadata.json", metadata)

    report = SACMetaTrainingReport(
        total_timesteps=config.total_timesteps,
        train_rows=int(train_context["prices"].shape[0]),
        eval_rows=int(eval_context["prices"].shape[0]),
        eval_mean_reward=evaluation["eval_mean_reward"],
        eval_mean_final_value=evaluation["eval_mean_final_value"],
        eval_mean_sharpe=evaluation["eval_mean_sharpe"],
        eval_mean_max_drawdown=evaluation["eval_mean_max_drawdown"],
        eval_mean_agent_alpha=evaluation["eval_mean_agent_alpha"],
        model_path=str(final_model_path),
        backup_path=str(backup_path) if backup_path is not None else None,
        trained_at=datetime.now(UTC).isoformat(),
    )
    _save_json(staging_meta_dir / "training_summary.json", asdict(report))
    _commit_staged_artifact_dir(staging_meta_dir, meta_dir)
    return report
