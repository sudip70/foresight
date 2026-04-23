from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import importlib.util
import math

import numpy as np

from backend.app.core.config import Settings
from backend.app.ml.artifacts import (
    ASSET_CLASSES,
    AssetArtifacts,
    ArtifactValidationError,
    MetaArtifacts,
    load_asset_artifacts,
    load_meta_artifacts,
)
from backend.app.ml.envs import MetaPortfolioEnv, SingleAgentEnv, meta_observation_dim
from backend.app.ml.feature_groups import FeatureSlices, build_feature_slices
from backend.app.ml.policies import apply_class_guardrails, normalize_weights


MACRO_FEATURE_NAMES = [
    "vix_market_volatility",
    "federal_funds_rate",
    "ten_year_treasury_yield",
    "unemployment_rate",
    "cpi_all_items",
    "recession_indicator",
]


@dataclass
class InferenceSummary:
    expected_daily_return: float
    annualized_return: float
    portfolio_variance: float
    annualized_volatility: float


@dataclass
class InferenceResult:
    summary: InferenceSummary
    asset_allocations: list[dict]
    class_allocations: list[dict]
    sub_agent_allocations: dict[str, list[dict]]
    latest_snapshot: dict
    warnings: list[str]
    model_version: str
    top_asset_targets: list[str]
    feature_slices: FeatureSlices
    latest_observation: np.ndarray


@dataclass
class BacktestResult:
    summary_metrics: dict[str, float]
    equity_curve: list[dict]
    drawdown_curve: list[dict]
    warnings: list[str]


@dataclass
class RuntimeBundle:
    assets: dict[str, AssetArtifacts]
    meta: MetaArtifacts


def _dependency_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _block_diagonal(blocks: list[np.ndarray]) -> np.ndarray:
    total = sum(block.shape[0] for block in blocks)
    matrix = np.zeros((total, total), dtype=float)
    cursor = 0
    for block in blocks:
        width = block.shape[0]
        matrix[cursor : cursor + width, cursor : cursor + width] = block
        cursor += width
    return matrix


def _mode_regimes(regime_stack: np.ndarray) -> np.ndarray:
    result = []
    for column in regime_stack.T:
        counts = np.bincount(column.astype(int), minlength=3)
        result.append(int(np.argmax(counts)))
    return np.asarray(result, dtype=int)


def _common_date_positions(dates: np.ndarray, common_dates: np.ndarray) -> np.ndarray:
    dates = np.asarray(dates)
    order = np.argsort(dates)
    sorted_dates = dates[order]
    positions = np.searchsorted(sorted_dates, common_dates)
    if (
        np.any(positions >= sorted_dates.shape[0])
        or not np.array_equal(sorted_dates[positions], common_dates)
    ):
        raise ArtifactValidationError("Unable to align artifacts by common dates")
    return order[positions]


class StockifyEngine:
    def __init__(self, settings: Settings, *, strict_validation: bool = False) -> None:
        self.settings = settings
        self.strict_validation = strict_validation
        assets = {
            asset_class: load_asset_artifacts(
                settings.artifact_root, asset_class, strict=strict_validation
            )
            for asset_class in ASSET_CLASSES
        }
        self._combined_context = self._build_combined_context(assets)
        total_assets = sum(len(bundle.tickers) for bundle in assets.values())
        meta_obs_dim = meta_observation_dim(
            n_assets=total_assets,
            micro_dim=int(self._combined_context["combined_micro"].shape[1]),
            macro_dim=int(self._combined_context["combined_macro"].shape[1]),
        )
        meta = load_meta_artifacts(
            settings.artifact_root,
            observation_dim=meta_obs_dim,
            action_dim=total_assets,
        )
        self.runtime = RuntimeBundle(assets=assets, meta=meta)

    def _build_combined_context(self, assets: dict[str, AssetArtifacts]) -> dict:
        date_arrays = [np.asarray(bundle.dates) for bundle in assets.values()]
        common_dates = date_arrays[0]
        for dates in date_arrays[1:]:
            common_dates = np.intersect1d(common_dates, dates)
        if common_dates.shape[0] < 2:
            raise ArtifactValidationError("Not enough common dates across asset classes")

        aligned_assets: dict[str, dict] = {}
        warnings: list[str] = []
        for asset_class, bundle in assets.items():
            date_positions = _common_date_positions(bundle.dates, common_dates)
            prices = bundle.prices[date_positions]
            ohlcv = bundle.ohlcv[date_positions]
            regimes = bundle.regimes[date_positions]
            micro = bundle.micro_indicators[date_positions]
            macro = bundle.macro_indicators[date_positions]
            aligned_assets[asset_class] = {
                "dates": common_dates,
                "prices": prices,
                "ohlcv": ohlcv,
                "regimes": regimes,
                "micro": micro,
                "macro": macro,
            }
            if bundle.alignment.trimmed:
                warnings.append(
                    f"{asset_class} artifacts were trimmed from "
                    f"{bundle.alignment.original_rows} to {bundle.alignment.aligned_rows} aligned rows"
                )
            if common_dates.shape[0] != bundle.alignment.aligned_rows:
                warnings.append(
                    f"{asset_class} artifacts were date-aligned from "
                    f"{bundle.alignment.aligned_rows} to {common_dates.shape[0]} common rows"
                )

        combined_micro = np.hstack([aligned_assets[name]["micro"] for name in ASSET_CLASSES])
        combined_macro = np.hstack([aligned_assets[name]["macro"] for name in ASSET_CLASSES])
        combined_regimes = _mode_regimes(
            np.vstack([aligned_assets[name]["regimes"] for name in ASSET_CLASSES])
        )
        combined_prices = np.hstack([aligned_assets[name]["prices"] for name in ASSET_CLASSES])

        return {
            "aligned_rows": int(common_dates.shape[0]),
            "dates": common_dates,
            "assets": aligned_assets,
            "combined_micro": combined_micro,
            "combined_macro": combined_macro,
            "combined_regimes": combined_regimes,
            "combined_prices": combined_prices,
            "warnings": warnings,
        }

    def dependency_status(self) -> dict[str, bool]:
        return {
            "numpy": _dependency_installed("numpy"),
            "pandas": _dependency_installed("pandas"),
            "scikit_learn": _dependency_installed("sklearn"),
            "shap": _dependency_installed("shap"),
            "stable_baselines3": _dependency_installed("stable_baselines3"),
            "gymnasium": _dependency_installed("gymnasium"),
        }

    def health_payload(self) -> dict:
        return {
            "status": "ok",
            "artifact_root": str(self.settings.artifact_root),
            "dataset_root": str(self.settings.dataset_root),
            "combined_aligned_rows": self._combined_context["aligned_rows"],
            "combined_date_start": str(self._combined_context["dates"][0]),
            "combined_date_end": str(self._combined_context["dates"][-1]),
            "dependencies": self.dependency_status(),
            "artifacts": {
                asset_class: {
                    "aligned_rows": bundle.alignment.aligned_rows,
                    "trimmed": bundle.alignment.trimmed,
                    "feature_version": bundle.metadata.get("feature_version"),
                }
                for asset_class, bundle in self.runtime.assets.items()
            },
            "meta_model": {
                "feature_version": self.runtime.meta.metadata.get("feature_version"),
                "policy_backend": self.runtime.meta.metadata.get("policy_backend"),
            },
        }

    def model_payload(self) -> dict:
        asset_classes = []
        for asset_class, bundle in self.runtime.assets.items():
            asset_classes.append(
                {
                    "asset_class": asset_class,
                    "tickers": bundle.tickers,
                    "aligned_rows": bundle.alignment.aligned_rows,
                    "trimmed": bundle.alignment.trimmed,
                    "feature_version": bundle.metadata.get("feature_version"),
                    "policy_backend": bundle.metadata.get("policy_backend"),
                    "algorithm": bundle.metadata.get("algorithm"),
                }
            )

        slices = build_feature_slices(
            n_assets=sum(len(bundle.tickers) for bundle in self.runtime.assets.values()),
            micro_dim=self._combined_context["combined_micro"].shape[1],
            macro_dim=self._combined_context["combined_macro"].shape[1],
        )
        feature_groups = {
            name: {"start": data_slice.start, "stop": data_slice.stop}
            for name, data_slice in slices.as_dict().items()
        }
        return {
            "supported_asset_classes": list(ASSET_CLASSES),
            "asset_agents": asset_classes,
            "meta_agent": {
                "algorithm": self.runtime.meta.metadata.get("algorithm"),
                "policy_backend": self.runtime.meta.metadata.get("policy_backend"),
                "feature_version": self.runtime.meta.metadata.get("feature_version"),
            },
            "feature_groups": feature_groups,
            "explainability": {
                "method": "surrogate-shap",
                "grouping": list(feature_groups.keys()),
                "top_asset_target_count": self.settings.top_asset_target_count,
            },
        }

    def _compute_mu_cov(self, prices: np.ndarray, window_size: int, step: int) -> tuple[np.ndarray, np.ndarray]:
        start = max(0, step - window_size)
        window_prices = prices[start : step + 1]
        if window_prices.shape[0] < 2:
            returns = np.zeros((1, prices.shape[1]), dtype=float)
        else:
            returns = np.diff(np.log(np.clip(window_prices, 1e-12, None)), axis=0)
        mu = np.mean(returns, axis=0)
        cov = (
            np.cov(returns.T)
            if returns.shape[0] > 1
            else np.zeros((prices.shape[1], prices.shape[1]), dtype=float)
        )
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=float)
        cov += np.eye(cov.shape[0]) * 1e-6
        return mu.astype(float), cov.astype(float)

    def _class_ranges(self) -> dict[str, tuple[int, int]]:
        cursor = 0
        ranges = {}
        for asset_class in ASSET_CLASSES:
            width = len(self.runtime.assets[asset_class].tickers)
            ranges[asset_class] = (cursor, cursor + width)
            cursor += width
        return ranges

    def _sub_agent_max_weight(self, bundle: AssetArtifacts) -> float | None:
        config = bundle.metadata.get("ppo_training_config", {})
        max_weight = config.get("max_asset_weight")
        return None if max_weight is None else float(max_weight)

    def _meta_max_class_weights(self) -> dict[str, float]:
        return {
            "stock": self.settings.meta_max_stock_weight,
            "crypto": self.settings.meta_max_crypto_weight,
            "etf": self.settings.meta_max_etf_weight,
        }

    def _apply_risk_adjustments(self, weights: np.ndarray, risk: float) -> np.ndarray:
        return apply_class_guardrails(
            weights,
            class_ranges=self._class_ranges(),
            max_class_weights=self._meta_max_class_weights(),
            max_asset_weight=self.settings.meta_max_asset_weight,
        )

    def _build_class_features(
        self,
        *,
        sub_agent_weights: np.ndarray,
        mu_all: np.ndarray,
        cov_all: np.ndarray,
        prev_weights: np.ndarray,
    ) -> np.ndarray:
        features = []
        for asset_class, (start, end) in self._class_ranges().items():
            class_weights = normalize_weights(sub_agent_weights[start:end])
            class_mu = mu_all[start:end]
            class_cov = cov_all[start:end, start:end]
            class_expected_return = float(np.dot(class_weights, class_mu))
            class_volatility = float(np.sqrt(max(class_weights.T @ class_cov @ class_weights, 0.0)))
            class_prev_weight = float(prev_weights[start:end].sum())
            features.extend([class_expected_return, class_volatility, class_prev_weight])
        return np.asarray(features, dtype=np.float32)

    def _build_snapshot(self, step: int, sub_agent_allocations: dict[str, np.ndarray]) -> dict:
        stock_macro = self._combined_context["assets"]["stock"]["macro"][step]
        macro_names = self.runtime.assets["stock"].metadata.get(
            "macro_feature_names", MACRO_FEATURE_NAMES
        )
        macro_snapshot = [
            {"name": name, "normalized_value": float(stock_macro[index])}
            for index, name in enumerate(macro_names[: len(stock_macro)])
        ]
        regime_value = int(self._combined_context["combined_regimes"][step])
        class_ranges = self._class_ranges()
        return {
            "date": str(self._combined_context["dates"][step]),
            "global_regime": regime_value,
            "macro": macro_snapshot,
            "class_ranges": class_ranges,
            "sub_agent_weight_sums": {
                asset_class: float(weights.sum())
                for asset_class, weights in sub_agent_allocations.items()
            },
        }

    def _build_step_scenario(
        self,
        *,
        step: int,
        risk: float,
        window_size: int,
        prev_weights: np.ndarray | None,
        prev_sub_weights: dict[str, np.ndarray] | None = None,
    ) -> dict:
        asset_outputs = {}
        mu_blocks = []
        cov_blocks = []
        concatenated_sub_weights = []

        for asset_class in ASSET_CLASSES:
            bundle = self.runtime.assets[asset_class]
            asset_context = self._combined_context["assets"][asset_class]
            env = SingleAgentEnv(
                prices=asset_context["prices"],
                ohlcv=asset_context["ohlcv"],
                regimes=asset_context["regimes"],
                micro_indicators=asset_context["micro"],
                macro_indicators=asset_context["macro"],
                risk_appetite=risk,
            )
            sub_prev_weights = (
                prev_sub_weights.get(asset_class) if prev_sub_weights is not None else None
            )
            observation = env.observation_at(step, prev_weights=sub_prev_weights)
            weights = normalize_weights(
                bundle.policy.predict(observation),
                max_weight=self._sub_agent_max_weight(bundle),
            )
            mu, cov = self._compute_mu_cov(asset_context["prices"], window_size, step)
            asset_outputs[asset_class] = {"weights": weights, "mu": mu, "cov": cov}
            mu_blocks.append(mu)
            cov_blocks.append(cov)
            concatenated_sub_weights.append(weights)

        mu_all = np.concatenate(mu_blocks)
        cov_all = _block_diagonal(cov_blocks)
        sub_agent_weights = normalize_weights(np.concatenate(concatenated_sub_weights))
        base_prev_weights = (
            sub_agent_weights
            if prev_weights is None
            else normalize_weights(prev_weights)
        )
        class_features = self._build_class_features(
            sub_agent_weights=sub_agent_weights,
            mu_all=mu_all,
            cov_all=cov_all,
            prev_weights=base_prev_weights,
        )

        meta_env = MetaPortfolioEnv(
            micro_indicators=self._combined_context["combined_micro"],
            macro_indicators=self._combined_context["combined_macro"],
            regimes=self._combined_context["combined_regimes"],
            n_assets=len(mu_all),
        )
        meta_observation = meta_env.observation_at(
            step,
            mu=mu_all,
            cov_diag=np.diag(cov_all),
            prev_weights=base_prev_weights,
            sub_agent_weights=sub_agent_weights,
            class_features=class_features,
            risk_appetite=risk,
            portfolio_value_ratio=1.0,
        )

        return {
            "asset_outputs": asset_outputs,
            "mu_all": mu_all,
            "cov_all": cov_all,
            "sub_agent_weights": sub_agent_weights,
            "class_features": class_features,
            "meta_observation": meta_observation,
            "prev_weights": base_prev_weights,
        }

    def run_inference(
        self,
        *,
        amount: float,
        risk: float,
        duration: int,
        window_size: int,
    ) -> InferenceResult:
        step = self._combined_context["aligned_rows"] - 1
        scenario = self._build_step_scenario(
            step=step,
            risk=risk,
            window_size=window_size,
            prev_weights=None,
        )
        raw_weights = self.runtime.meta.policy.predict(scenario["meta_observation"])
        final_weights = self._apply_risk_adjustments(raw_weights, risk)
        asset_names = []
        for asset_class in ASSET_CLASSES:
            asset_names.extend(
                [(asset_class, ticker) for ticker in self.runtime.assets[asset_class].tickers]
            )

        class_ranges = self._class_ranges()
        asset_allocations = []
        for index, (asset_class, ticker) in enumerate(asset_names):
            asset_allocations.append(
                {
                    "asset_class": asset_class,
                    "ticker": ticker,
                    "weight": float(final_weights[index]),
                    "amount": float(final_weights[index] * amount),
                }
            )

        class_allocations = []
        for asset_class, (start, end) in class_ranges.items():
            class_weight = float(final_weights[start:end].sum())
            class_allocations.append(
                {
                    "asset_class": asset_class,
                    "weight": class_weight,
                    "amount": float(class_weight * amount),
                }
            )

        sub_agent_allocations = {}
        for asset_class in ASSET_CLASSES:
            weights = scenario["asset_outputs"][asset_class]["weights"]
            sub_agent_allocations[asset_class] = [
                {
                    "asset_class": asset_class,
                    "ticker": ticker,
                    "weight": float(weight),
                    "amount": float(weight * amount),
                }
                for ticker, weight in zip(self.runtime.assets[asset_class].tickers, weights)
            ]

        expected_daily_return = float(np.dot(final_weights, scenario["mu_all"]))
        annualized_return = float(max((1 + expected_daily_return) ** 252 - 1, -1.0))
        portfolio_variance = float(final_weights.T @ scenario["cov_all"] @ final_weights)
        annualized_volatility = float(math.sqrt(max(portfolio_variance, 0.0) * 252))

        top_asset_targets = [
            allocation["ticker"]
            for allocation in sorted(
                asset_allocations, key=lambda entry: entry["weight"], reverse=True
            )[: self.settings.top_asset_target_count]
        ]

        feature_slices = build_feature_slices(
            n_assets=len(asset_names),
            micro_dim=self._combined_context["combined_micro"].shape[1],
            macro_dim=self._combined_context["combined_macro"].shape[1],
        )

        return InferenceResult(
            summary=InferenceSummary(
                expected_daily_return=expected_daily_return,
                annualized_return=annualized_return,
                portfolio_variance=portfolio_variance,
                annualized_volatility=annualized_volatility,
            ),
            asset_allocations=asset_allocations,
            class_allocations=class_allocations,
            sub_agent_allocations=sub_agent_allocations,
            latest_snapshot=self._build_snapshot(
                step,
                {
                    asset_class: scenario["asset_outputs"][asset_class]["weights"]
                    for asset_class in ASSET_CLASSES
                },
            ),
            warnings=list(self._combined_context["warnings"]),
            model_version=str(self.runtime.meta.metadata.get("feature_version", "unknown")),
            top_asset_targets=top_asset_targets,
            feature_slices=feature_slices,
            latest_observation=scenario["meta_observation"],
        )

    def run_backtest(
        self,
        *,
        initial_amount: float,
        risk: float,
        window_size: int,
        max_steps: int,
    ) -> BacktestResult:
        prices_all = self._combined_context["combined_prices"]
        max_available_steps = prices_all.shape[0] - window_size - 1
        steps_to_run = min(max_available_steps, max_steps)
        if steps_to_run <= 0:
            raise ArtifactValidationError("Not enough aligned rows to run a backtest")

        portfolio_value = float(initial_amount)
        prev_weights = None
        prev_sub_weights = None
        equity_curve = [{"step": 0, "value": portfolio_value}]
        drawdown_curve = [{"step": 0, "value": 0.0}]
        returns = []
        peak = portfolio_value

        start_step = window_size
        for offset, step in enumerate(range(start_step, start_step + steps_to_run), start=1):
            scenario = self._build_step_scenario(
                step=step,
                risk=risk,
                window_size=window_size,
                prev_weights=prev_weights,
                prev_sub_weights=prev_sub_weights,
            )
            weights = self._apply_risk_adjustments(
                self.runtime.meta.policy.predict(scenario["meta_observation"]),
                risk,
            )
            actual_returns = (
                prices_all[step + 1] - prices_all[step]
            ) / np.clip(prices_all[step], 1e-12, None)
            daily_return = float(np.dot(weights, actual_returns))
            portfolio_value *= 1 + daily_return
            peak = max(peak, portfolio_value)
            drawdown = 1 - (portfolio_value / peak)
            equity_curve.append({"step": offset, "value": float(portfolio_value)})
            drawdown_curve.append({"step": offset, "value": float(drawdown)})
            returns.append(daily_return)
            prev_weights = weights
            prev_sub_weights = {
                asset_class: scenario["asset_outputs"][asset_class]["weights"]
                for asset_class in ASSET_CLASSES
            }

        cumulative_return = float((portfolio_value / initial_amount) - 1)
        mean_daily_return = float(np.mean(returns))
        daily_volatility = float(np.std(returns))
        annualized_return = float((1 + mean_daily_return) ** 252 - 1)
        sharpe_ratio = (
            float((mean_daily_return / daily_volatility) * math.sqrt(252))
            if daily_volatility > 0
            else 0.0
        )
        max_drawdown = float(max(point["value"] for point in drawdown_curve))

        return BacktestResult(
            summary_metrics={
                "cumulative_return": cumulative_return,
                "annualized_return": annualized_return,
                "daily_volatility": daily_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "ending_value": float(portfolio_value),
            },
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            warnings=list(self._combined_context["warnings"]),
        )


@lru_cache(maxsize=4)
def get_engine(
    artifact_root: str,
    dataset_root: str,
    strict_validation: bool,
    surrogate_sample_size: int,
    surrogate_fidelity_threshold: float,
    top_asset_target_count: int,
    default_backtest_steps: int,
    meta_max_asset_weight: float,
    meta_max_stock_weight: float,
    meta_max_crypto_weight: float,
    meta_max_etf_weight: float,
) -> StockifyEngine:
    settings = Settings(
        project_name="Stockify Backend",
        api_prefix="/api",
        artifact_root=Path(artifact_root),
        dataset_root=Path(dataset_root),
        surrogate_sample_size=surrogate_sample_size,
        surrogate_fidelity_threshold=surrogate_fidelity_threshold,
        top_asset_target_count=top_asset_target_count,
        default_backtest_steps=default_backtest_steps,
        meta_max_asset_weight=meta_max_asset_weight,
        meta_max_stock_weight=meta_max_stock_weight,
        meta_max_crypto_weight=meta_max_crypto_weight,
        meta_max_etf_weight=meta_max_etf_weight,
    )
    return StockifyEngine(settings, strict_validation=strict_validation)
