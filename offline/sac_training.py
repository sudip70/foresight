from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
import shutil
import tempfile

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.app.ml.artifacts import ASSET_CLASSES, AssetArtifacts, load_asset_artifacts
from backend.app.ml.envs import (
    MetaPortfolioEnv,
    SingleAgentEnv,
    build_meta_observation,
    meta_observation_dim,
)
from backend.app.ml.policies import apply_class_guardrails, normalize_weights


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
    target_daily_volatility: float | None = 0.014
    target_volatility_penalty_scale: float = 0.25
    reward_scale: float = 100.0
    max_asset_weight: float = 0.20
    max_stock_weight: float = 0.85
    max_crypto_weight: float = 0.30
    max_etf_weight: float = 0.70
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
    return ranges


def _sub_agent_max_weight(bundle: AssetArtifacts) -> float | None:
    config = bundle.metadata.get("ppo_training_config", {})
    max_weight = config.get("max_asset_weight")
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


def _build_context(artifact_root: Path) -> dict:
    assets = {
        asset_class: load_asset_artifacts(artifact_root, asset_class, strict=False)
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
            "ohlcv": bundle.ohlcv[positions],
            "regimes": bundle.regimes[positions],
            "micro": bundle.micro_indicators[positions],
            "macro": bundle.macro_indicators[positions],
        }

    return {
        "assets": assets,
        "aligned_assets": aligned_assets,
        "dates": common_dates,
        "prices": np.hstack([aligned_assets[name]["prices"] for name in ASSET_CLASSES]),
        "micro": np.hstack([aligned_assets[name]["micro"] for name in ASSET_CLASSES]),
        "macro": np.hstack([aligned_assets[name]["macro"] for name in ASSET_CLASSES]),
        "regimes": _mode_regimes(
            np.vstack([aligned_assets[name]["regimes"] for name in ASSET_CLASSES])
        ),
        "class_ranges": _class_ranges(assets),
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


def _precompute_sub_agent_weights(context: dict, config: SACMetaTrainingConfig) -> dict[float, np.ndarray]:
    outputs: dict[float, np.ndarray] = {}
    rows = context["prices"].shape[0]
    for risk in config.risk_values:
        previous_by_class: dict[str, np.ndarray] | None = None
        risk_weights = np.zeros_like(context["prices"], dtype=float)
        for step in range(rows):
            class_weights = []
            next_previous: dict[str, np.ndarray] = {}
            for asset_class in ASSET_CLASSES:
                bundle = context["assets"][asset_class]
                asset_context = context["aligned_assets"][asset_class]
                env = SingleAgentEnv(
                    prices=asset_context["prices"],
                    ohlcv=asset_context["ohlcv"],
                    regimes=asset_context["regimes"],
                    micro_indicators=asset_context["micro"],
                    macro_indicators=asset_context["macro"],
                    risk_appetite=risk,
                )
                previous = (
                    previous_by_class.get(asset_class)
                    if previous_by_class is not None
                    else None
                )
                observation = env.observation_at(step, prev_weights=previous)
                weights = normalize_weights(
                    bundle.policy.predict(observation),
                    max_weight=_sub_agent_max_weight(bundle),
                )
                class_weights.append(weights)
                next_previous[asset_class] = weights
            previous_by_class = next_previous
            risk_weights[step] = normalize_weights(np.concatenate(class_weights))
        outputs[float(risk)] = risk_weights
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
        self.max_class_weights = {
            "stock": config.max_stock_weight,
            "crypto": config.max_crypto_weight,
            "etf": config.max_etf_weight,
        }
        self.mu, self.cov_diag = _compute_mu_cov_diag(self.prices, config.window_size)
        self.sub_agent_weights_by_risk = _precompute_sub_agent_weights(context, config)
        self.risk_values = tuple(float(value) for value in config.risk_values)
        self.n_assets = int(self.prices.shape[1])
        self.max_start = self.prices.shape[0] - config.episode_length - 2
        if self.max_start < config.window_size:
            raise ValueError("Not enough rows to train SAC meta env")

        obs_dim = meta_observation_dim(
            n_assets=self.n_assets,
            micro_dim=self.micro.shape[1],
            macro_dim=self.macro.shape[1],
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
            shape=(self.n_assets,),
            dtype=np.float32,
        )
        self.current_step = config.window_size
        self.end_step = self.current_step + config.episode_length
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.prev_weights = np.ones(self.n_assets, dtype=float) / self.n_assets
        self.risk_appetite = 0.5

    def _sample_risk(self) -> float:
        if self.fixed_risk is not None:
            return float(self.fixed_risk)
        index = int(self.np_random.integers(0, len(self.risk_values)))
        return self.risk_values[index]

    def _risk_key(self) -> float:
        return min(self.risk_values, key=lambda value: abs(value - self.risk_appetite))

    def _sub_agent_weights(self) -> np.ndarray:
        return self.sub_agent_weights_by_risk[self._risk_key()][self.current_step]

    def _class_features(self, sub_agent_weights: np.ndarray) -> np.ndarray:
        features = []
        for _, (start, end) in self.class_ranges.items():
            class_weights = normalize_weights(sub_agent_weights[start:end])
            class_mu = self.mu[self.current_step, start:end]
            class_cov_diag = self.cov_diag[self.current_step, start:end]
            class_expected_return = float(np.dot(class_weights, class_mu))
            class_volatility = float(np.sqrt(max(np.dot(class_weights**2, class_cov_diag), 0.0)))
            class_prev_weight = float(self.prev_weights[start:end].sum())
            features.extend([class_expected_return, class_volatility, class_prev_weight])
        return np.asarray(features, dtype=np.float32)

    def _observation(self) -> np.ndarray:
        sub_agent_weights = self._sub_agent_weights()
        return build_meta_observation(
            step=self.current_step,
            mu=self.mu[self.current_step],
            cov_diag=self.cov_diag[self.current_step],
            prev_weights=self.prev_weights,
            sub_agent_weights=sub_agent_weights,
            class_features=self._class_features(sub_agent_weights),
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
        self.prev_weights = self._sub_agent_weights()
        return self._observation(), {}

    def step(self, action):
        sub_agent_weights = self._sub_agent_weights()
        weights = apply_class_guardrails(
            action,
            class_ranges=self.class_ranges,
            max_class_weights=self.max_class_weights,
            max_asset_weight=self.config.max_asset_weight,
        )
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
        target_volatility_penalty = 0.0
        if (
            self.config.target_daily_volatility is not None
            and self.config.target_volatility_penalty_scale > 0
        ):
            target_volatility_penalty = self.config.target_volatility_penalty_scale * abs(
                portfolio_volatility - self.config.target_daily_volatility
            )
        raw_reward = (
            float(np.log(net_return))
            + self.config.benchmark_reward_scale * (portfolio_return - agent_benchmark_return)
            - self.config.volatility_penalty_scale * portfolio_volatility
            - self.config.turnover_penalty * turnover
            - self.config.concentration_penalty_scale * concentration
            - self.config.downside_penalty_scale * downside_return
            - self.config.drawdown_penalty_scale * (drawdown + drawdown_worsening)
            - target_volatility_penalty
        )
        reward = self.config.reward_scale * raw_reward

        self.portfolio_value = next_portfolio_value
        self.peak_value = next_peak
        self.prev_weights = weights
        self.current_step += 1
        terminated = self.current_step >= self.end_step
        return self._observation(), float(reward), terminated, False, {
            "portfolio_value": float(self.portfolio_value),
            "portfolio_return": portfolio_return,
            "agent_benchmark_return": agent_benchmark_return,
            "agent_alpha": portfolio_return - agent_benchmark_return,
            "portfolio_volatility": portfolio_volatility,
            "turnover": turnover,
            "drawdown": drawdown,
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
    context = _build_context(artifact_root)
    train_context, eval_context = split_meta_context(context, config)

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

    with tempfile.TemporaryDirectory(prefix="stockify_sac_meta_") as tmp_dir:
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
    model_path = meta_dir / "model.zip"
    backup_path = None
    if model_path.exists():
        backup_path = meta_dir / (
            f"model.backup_before_sac_retrain_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.zip"
        )
        shutil.copy2(model_path, backup_path)
    trained_model.save(model_path)

    obs_dim = meta_observation_dim(
        n_assets=context["prices"].shape[1],
        micro_dim=context["micro"].shape[1],
        macro_dim=context["macro"].shape[1],
    )
    metadata = {
        "algorithm": "sac",
        "policy_backend": "sb3",
        "model_file": "model.zip",
        "feature_version": "sac-meta-v2-subagent-aware",
        "policy_observation_dim": int(obs_dim),
        "inference_observation_dim": int(obs_dim),
        "action_dim": int(context["prices"].shape[1]),
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
            "hard_min_stock_weight_removed": True,
        },
        "sac_training_config": asdict(config),
        "sac_retrained_at": datetime.now(UTC).isoformat(),
        "train_rows": int(train_context["prices"].shape[0]),
        "eval_rows": int(eval_context["prices"].shape[0]),
        **evaluation,
    }
    _save_json(meta_dir / "metadata.json", metadata)

    report = SACMetaTrainingReport(
        total_timesteps=config.total_timesteps,
        train_rows=int(train_context["prices"].shape[0]),
        eval_rows=int(eval_context["prices"].shape[0]),
        eval_mean_reward=evaluation["eval_mean_reward"],
        eval_mean_final_value=evaluation["eval_mean_final_value"],
        eval_mean_sharpe=evaluation["eval_mean_sharpe"],
        eval_mean_max_drawdown=evaluation["eval_mean_max_drawdown"],
        eval_mean_agent_alpha=evaluation["eval_mean_agent_alpha"],
        model_path=str(model_path),
        backup_path=str(backup_path) if backup_path is not None else None,
        trained_at=datetime.now(UTC).isoformat(),
    )
    _save_json(meta_dir / "training_summary.json", asdict(report))
    return report
