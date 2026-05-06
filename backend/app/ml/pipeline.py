from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from types import MappingProxyType
import importlib.util
import math

import numpy as np

from backend.app.core.config import Settings
from backend.app.market.simulation import (
    allocate_simulation_risky_weights,
    select_diversified_simulation_forecasts,
)
from backend.app.ml.artifacts import (
    ASSET_CLASSES,
    AssetArtifacts,
    ArtifactValidationError,
    MetaArtifacts,
    load_asset_artifacts,
    load_meta_artifacts,
    peek_meta_metadata,
)
from backend.app.ml.envs import MetaPortfolioEnv, SingleAgentEnv, meta_observation_dim
from backend.app.ml.feature_groups import FeatureSlices, build_feature_slices
from backend.app.ml.policies import (
    MetaSignalPolicy,
    SingleAgentSignalPolicy,
    apply_class_guardrails,
    apply_cash_risk_managed_overlay,
    blend_allocation_sources,
    constrain_turnover,
    normalize_action_with_cash_sleeve,
    normalize_weights,
    normalize_weights_with_caps,
    policy_signal_blend_weight,
)


CASH_ASSET_CLASS = "cash"
CASH_TICKER = "CASH"
META_V3_PREFIX = "sac-meta-v3"


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
    projected_horizon_return: float
    projected_value: float
    projected_profit: float
    downside_value: float
    upside_value: float


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
    trade_log: list[dict]


@dataclass
class BacktestResult:
    summary_metrics: dict[str, float]
    equity_curve: list[dict]
    drawdown_curve: list[dict]
    trade_log: list[dict]
    warnings: list[str]


@dataclass
class RuntimeBundle:
    assets: dict[str, AssetArtifacts]
    meta: MetaArtifacts


@dataclass(frozen=True)
class ArtifactStore:
    settings: Settings
    strict_validation: bool = True

    def load_assets(self) -> dict[str, AssetArtifacts]:
        return {
            asset_class: load_asset_artifacts(
                self.settings.artifact_root,
                asset_class,
                strict=self.strict_validation,
                policy_mode=self.settings.artifact_policy_mode,
            )
            for asset_class in ASSET_CLASSES
        }

    def load_meta(self, *, observation_dim: int, action_dim: int) -> MetaArtifacts:
        return load_meta_artifacts(
            self.settings.artifact_root,
            observation_dim=observation_dim,
            action_dim=action_dim,
            policy_mode=self.settings.artifact_policy_mode,
        )


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


class ForesightEngine:
    def _meta_metadata(self) -> dict:
        if hasattr(self, "runtime"):
            return self.runtime.meta.metadata
        return self._meta_metadata_hint

    def _use_v3_meta_architecture(self) -> bool:
        return str(self._meta_metadata().get("feature_version", "")).startswith(META_V3_PREFIX)

    def _meta_class_feature_dim(self) -> int:
        default = 12 if self._use_v3_meta_architecture() else 9
        return int(self._meta_metadata().get("class_feature_dim", default))

    def _meta_macro_dim(self, context: dict) -> int:
        if self._use_v3_meta_architecture():
            return int(context["global_macro_raw"].shape[1])
        return int(context["combined_macro_legacy"].shape[1])

    def _resolve_meta_macro_matrix(self, context: dict) -> np.ndarray:
        if not self._use_v3_meta_architecture():
            return context["combined_macro_legacy"]
        matrix = np.asarray(context["global_macro_raw"], dtype=float)
        scaler = self.runtime.meta.macro_scaler if hasattr(self, "runtime") else None
        if scaler is None:
            return matrix
        transformed = scaler.transform(matrix)
        return np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(float)

    def __init__(self, settings: Settings, *, strict_validation: bool = True) -> None:
        self.settings = settings
        self.strict_validation = strict_validation
        self._meta_metadata_hint = peek_meta_metadata(settings.artifact_root)
        self.artifact_store = ArtifactStore(settings, strict_validation=strict_validation)
        assets = self.artifact_store.load_assets()
        combined_context = self._build_combined_context(assets)
        total_assets = sum(len(bundle.tickers) for bundle in assets.values()) + (
            1 if settings.meta_cash_enabled else 0
        )
        meta_obs_dim = meta_observation_dim(
            n_assets=total_assets,
            micro_dim=int(combined_context["combined_micro"].shape[1]),
            macro_dim=self._meta_macro_dim(combined_context),
            class_feature_dim=self._meta_class_feature_dim(),
        )
        use_trained_meta_action_layout = (
            self.settings.artifact_policy_mode != "signal"
            and self._meta_metadata_hint.get("policy_action_layout")
            == "stock_crypto_etf_cash_sleeves"
        )
        meta = self.artifact_store.load_meta(
            observation_dim=meta_obs_dim,
            action_dim=int(
                self._meta_metadata_hint.get("action_dim", total_assets)
                if use_trained_meta_action_layout
                else total_assets
            ),
        )
        self.runtime = RuntimeBundle(assets=assets, meta=meta)
        self._combined_context = MappingProxyType(
            {
                **combined_context,
                "combined_macro": self._resolve_meta_macro_matrix(combined_context),
            }
        )
        self._single_agent_signal_policies = {
            asset_class: SingleAgentSignalPolicy(
                action_dim=int(bundle.metadata.get("action_dim", len(bundle.tickers))),
                observation_dim=int(
                    bundle.metadata.get(
                        "inference_observation_dim",
                        bundle.metadata.get("policy_observation_dim"),
                    )
                ),
                cash_enabled=bool(bundle.metadata.get("cash_enabled", False)),
            )
            for asset_class, bundle in self.runtime.assets.items()
        }
        self._meta_signal_policy = MetaSignalPolicy(
            action_dim=total_assets,
            observation_dim=int(
                self.runtime.meta.metadata.get(
                    "inference_observation_dim",
                    self.runtime.meta.metadata.get("policy_observation_dim", meta_obs_dim),
                )
            ),
            class_feature_dim=self._meta_class_feature_dim(),
            class_ranges=self._class_ranges(),
            cash_enabled=self.settings.meta_cash_enabled,
        )
        self._all_ticker_forecast_cache: dict[tuple[int, int, str], list[dict]] = {}

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
            macro_raw = bundle.macro_indicators_raw[date_positions]
            aligned_assets[asset_class] = {
                "dates": common_dates,
                "prices": prices,
                "risky_prices": prices[:, : len(bundle.tickers)],
                "ohlcv": ohlcv,
                "regimes": regimes,
                "micro": micro,
                "macro": macro,
                "macro_raw": macro_raw,
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
        combined_macro_legacy = np.hstack(
            [aligned_assets[name]["macro"] for name in ASSET_CLASSES]
        )
        combined_regimes = _mode_regimes(
            np.vstack([aligned_assets[name]["regimes"] for name in ASSET_CLASSES])
        )
        combined_prices = np.hstack(
            [aligned_assets[name]["risky_prices"] for name in ASSET_CLASSES]
        )
        global_macro_source = max(
            ASSET_CLASSES,
            key=lambda name: aligned_assets[name]["macro_raw"].shape[1],
        )
        global_macro_raw = np.asarray(aligned_assets[global_macro_source]["macro_raw"], dtype=float)
        global_macro_feature_names = assets[global_macro_source].metadata.get(
            "raw_macro_feature_names",
            assets[global_macro_source].metadata.get("macro_feature_names", MACRO_FEATURE_NAMES),
        )

        return {
            "aligned_rows": int(common_dates.shape[0]),
            "dates": common_dates,
            "assets": aligned_assets,
            "combined_micro": combined_micro,
            "combined_macro_legacy": combined_macro_legacy,
            "combined_macro": combined_macro_legacy,
            "combined_regimes": combined_regimes,
            "combined_prices": combined_prices,
            "global_macro_raw": global_macro_raw,
            "global_macro_feature_names": global_macro_feature_names,
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
                "policy_mode": self.settings.artifact_policy_mode,
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
                    "policy_mode": bundle.metadata.get(
                        "policy_mode",
                        self.settings.artifact_policy_mode,
                    ),
                    "algorithm": bundle.metadata.get("algorithm"),
                }
            )

        slices = build_feature_slices(
            n_assets=len(self._asset_names()),
            micro_dim=self._combined_context["combined_micro"].shape[1],
            macro_dim=self._combined_context["combined_macro"].shape[1],
            class_feature_dim=self._meta_class_feature_dim(),
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
                "policy_mode": self.runtime.meta.metadata.get(
                    "policy_mode",
                    self.settings.artifact_policy_mode,
                ),
                "feature_version": self.runtime.meta.metadata.get("feature_version"),
                "cash_enabled": self.settings.meta_cash_enabled,
                "max_cash_weight": self.settings.meta_max_cash_weight,
                "min_expected_daily_return": self.settings.meta_min_expected_daily_return,
                "class_feature_dim": self._meta_class_feature_dim(),
                "uses_shared_macro": self._use_v3_meta_architecture(),
                "meta_signal_blend": self._meta_signal_blend(),
                "sub_agent_consensus_blend": self._meta_agent_ensemble_blend(),
                "meta_macro_feature_names": list(
                    self.runtime.meta.metadata.get(
                        "meta_macro_feature_names",
                        self._combined_context["global_macro_feature_names"],
                    )
                ),
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

    def _projection_windows(self, requested_window: int, step: int) -> list[int]:
        max_window = max(int(step), 1)
        candidates = [20, 60, 126, 252]
        if requested_window >= 20:
            candidates.append(int(requested_window))
        windows = sorted(
            {
                min(window, max_window)
                for window in candidates
                if min(window, max_window) >= 2
            }
        )
        if not windows and max_window >= 1:
            windows = [max_window]
        return windows

    def _return_caps_for_asset_class(self, asset_class: str) -> tuple[float, float]:
        if asset_class == "crypto":
            return -0.0035, 0.0025
        if asset_class == "stock":
            return -0.0020, 0.0015
        return -0.0016, 0.0012

    def _compute_projection_mu_cov(
        self,
        prices: np.ndarray,
        window_size: int,
        step: int,
        *,
        asset_class: str,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        windows = self._projection_windows(window_size, step)
        mus = []
        covs = []
        sample_weights = []
        for projection_window in windows:
            mu, cov = self._compute_mu_cov(prices, projection_window, step)
            mus.append(mu)
            covs.append(cov)
            sample_weights.append(float(projection_window))

        weights = np.asarray(sample_weights, dtype=float)
        weights = weights / max(float(weights.sum()), 1e-12)
        blended_mu = np.average(np.vstack(mus), axis=0, weights=weights)
        blended_cov = np.average(np.stack(covs, axis=0), axis=0, weights=weights)

        long_mu = mus[-1]
        cash_mu = np.full_like(blended_mu, self._cash_daily_return(), dtype=float)
        prior_mu = (0.65 * long_mu) + (0.35 * cash_mu)
        effective_window = max(windows) if windows else max(int(step), 1)
        shrink = 80.0 / (80.0 + float(effective_window))
        stabilized_mu = ((1.0 - shrink) * blended_mu) + (shrink * prior_mu)

        lower_cap, upper_cap = self._return_caps_for_asset_class(asset_class)
        stabilized_mu = np.clip(stabilized_mu, lower_cap, upper_cap)
        diagonal_cov = np.diag(np.diag(blended_cov))
        stabilized_cov = (0.80 * blended_cov) + (0.20 * diagonal_cov)
        stabilized_cov += np.eye(stabilized_cov.shape[0]) * 1e-6

        return stabilized_mu.astype(float), stabilized_cov.astype(float), {
            "method": "multi_window_shrunk",
            "requested_window": int(window_size),
            "projection_windows": [int(window) for window in windows],
            "shrinkage_to_prior": float(shrink),
            "lower_daily_return_cap": float(lower_cap),
            "upper_daily_return_cap": float(upper_cap),
        }

    def _ticker_lookup(self) -> dict[str, tuple[str, int]]:
        lookup: dict[str, tuple[str, int]] = {}
        for asset_class, bundle in self.runtime.assets.items():
            for index, ticker in enumerate(bundle.tickers):
                lookup[ticker.upper()] = (asset_class, index)
        return lookup

    def _resolve_ticker(self, ticker: str) -> tuple[str, int, str]:
        normalized = ticker.strip().upper()
        lookup = self._ticker_lookup()
        if normalized not in lookup:
            raise ArtifactValidationError(f"Unsupported ticker: {ticker}")
        asset_class, index = lookup[normalized]
        canonical = self.runtime.assets[asset_class].tickers[index]
        return asset_class, index, canonical

    def _historical_drawdown(self, prices: np.ndarray) -> float:
        series = np.asarray(prices, dtype=float).reshape(-1)
        if series.shape[0] <= 1:
            return 0.0
        running_peak = np.maximum.accumulate(series)
        drawdown = 1.0 - (series / np.clip(running_peak, 1e-12, None))
        return float(np.max(drawdown))

    def _confidence_label(self, confidence: float) -> str:
        if confidence >= 0.70:
            return "High"
        if confidence >= 0.45:
            return "Medium"
        return "Low"

    def _risk_label(self, annualized_volatility: float, max_drawdown: float) -> str:
        risk_score = annualized_volatility + (0.60 * max_drawdown)
        if risk_score >= 0.65:
            return "High"
        if risk_score >= 0.30:
            return "Moderate"
        return "Lower"

    def _public_scenario_return_caps(self, asset_class: str) -> tuple[float, float]:
        if asset_class == "crypto":
            return -0.0030, 0.00125
        if asset_class == "stock":
            return -0.0016, 0.00075
        return -0.0012, 0.00055

    def _soft_cap_return(self, value: float, *, lower: float, upper: float) -> float:
        if value >= 0.0:
            return float(upper * math.tanh(value / max(upper, 1e-12)))
        lower_abs = abs(lower)
        return float(-lower_abs * math.tanh(abs(value) / max(lower_abs, 1e-12)))

    def _public_scenario_daily_return(
        self,
        *,
        asset_class: str,
        price_series: np.ndarray,
        model_signal_mu: float,
    ) -> tuple[float, dict]:
        series = np.asarray(price_series, dtype=float).reshape(-1)
        log_prices = np.log(np.clip(series, 1e-12, None))
        long_mu = float((log_prices[-1] - log_prices[0]) / max(series.shape[0] - 1, 1))
        medium_start = max(series.shape[0] - 253, 0)
        medium_mu = float(
            (log_prices[-1] - log_prices[medium_start])
            / max(series.shape[0] - medium_start - 1, 1)
        )
        short_start = max(series.shape[0] - 64, 0)
        short_mu = float(
            (log_prices[-1] - log_prices[short_start])
            / max(series.shape[0] - short_start - 1, 1)
        )
        blended = (0.30 * model_signal_mu) + (0.45 * medium_mu) + (0.25 * long_mu)
        if model_signal_mu > 0.0 and medium_mu < 0.0:
            blended *= 0.70
        if model_signal_mu < 0.0 and medium_mu > 0.0:
            blended = (0.50 * blended) + (0.50 * short_mu)
        lower, upper = self._public_scenario_return_caps(asset_class)
        capped = self._soft_cap_return(blended, lower=lower, upper=upper)
        return capped, {
            "model_signal_daily_return": float(model_signal_mu),
            "public_scenario_daily_return": float(capped),
            "long_run_daily_return": long_mu,
            "medium_daily_return": medium_mu,
            "short_daily_return": short_mu,
            "public_lower_daily_return_cap": lower,
            "public_upper_daily_return_cap": upper,
        }

    def _scenario_path(
        self,
        *,
        latest_price: float,
        daily_mu: float,
        daily_volatility: float,
        horizon_days: int,
        latest_date: np.datetime64,
        forecast_start_date: str,
        z_score: float,
        price_series: np.ndarray,
    ) -> list[dict]:
        horizon = max(int(horizon_days), 1)
        days = list(range(0, horizon + 1))
        forecast_start = np.datetime64(forecast_start_date, "D")
        historical = np.asarray(price_series, dtype=float).reshape(-1)
        recent_window = min(max(horizon, 30), 90, max(historical.shape[0] - 1, 1))
        recent_prices = historical[-(recent_window + 1) :]
        recent_returns = np.diff(np.log(np.clip(recent_prices, 1e-12, None)))
        if recent_returns.shape[0] == 0:
            residuals = np.zeros(horizon, dtype=float)
        else:
            residuals = recent_returns - float(np.mean(recent_returns))
            if float(np.std(residuals)) > 1e-12 and daily_volatility > 1e-12:
                residuals = residuals * (daily_volatility / float(np.std(residuals)))
            seed = (
                int(abs(float(latest_price)) * 10_000)
                ^ (horizon * 1_000_003)
                ^ int((float(z_score) + 3.0) * 100_000)
                ^ int(np.datetime64(latest_date, "D").astype(int))
            )
            rng = np.random.default_rng(seed)
            residuals = rng.choice(residuals, size=horizon, replace=True)

        cumulative_noise = np.concatenate(
            [np.array([0.0], dtype=float), np.cumsum(residuals)]
        )
        # Bridge the residual sequence back to zero so the final scenario target is preserved.
        bridge = cumulative_noise - (
            (np.asarray(days, dtype=float) / float(horizon)) * cumulative_noise[-1]
        )
        noise_scale = 0.32 if abs(z_score) < 1e-12 else 0.42
        path = []
        for index, day in enumerate(days):
            projected = latest_price * math.exp(
                (daily_mu * day)
                + (z_score * daily_volatility * math.sqrt(max(day, 0)))
                + (noise_scale * bridge[index])
            )
            path.append(
                {
                    "day": int(day),
                    "date": str(forecast_start + np.timedelta64(int(day), "D")),
                    "price": float(projected),
                }
            )
        return path

    def _forecast_score(
        self,
        forecast: dict,
        *,
        risk: float,
    ) -> float:
        risk_value = float(np.clip(risk, 0.0, 1.0))
        base_return = float(forecast["returns"]["base"])
        bull_return = float(forecast["returns"]["bull"])
        bear_return = float(forecast["returns"]["bear"])
        volatility = float(forecast["risk_metrics"]["annualized_volatility"])
        confidence = float(forecast["confidence"])
        downside = max(-bear_return, 0.0)
        return float(
            base_return
            + (risk_value * 0.35 * max(bull_return, 0.0))
            - ((1.0 - risk_value) * 0.65 * downside)
            - ((0.15 + (0.20 * (1.0 - risk_value))) * volatility)
            + (0.12 * confidence)
        )

    def _ticker_forecast_payload(
        self,
        *,
        ticker: str,
        horizon_days: int,
        window_size: int,
        forecast_start_date: str | None = None,
    ) -> dict:
        asset_class, index, canonical = self._resolve_ticker(ticker)
        asset_context = self._combined_context["assets"][asset_class]
        prices = np.asarray(asset_context["risky_prices"], dtype=float)
        price_series = prices[:, index]
        dates = self._combined_context["dates"]
        step = self._combined_context["aligned_rows"] - 1
        horizon = max(int(horizon_days), 1)

        projection_mu, projection_cov, estimator = self._compute_projection_mu_cov(
            prices,
            window_size,
            step,
            asset_class=asset_class,
        )
        model_signal_mu = float(projection_mu[index])
        daily_mu, public_estimator = self._public_scenario_daily_return(
            asset_class=asset_class,
            price_series=price_series,
            model_signal_mu=model_signal_mu,
        )
        daily_volatility = float(math.sqrt(max(float(projection_cov[index, index]), 0.0)))
        latest_price = float(price_series[-1])
        latest_date = dates[-1]
        forecast_start_date = forecast_start_date or date.today().isoformat()
        scenario_z = 0.84
        bear_path = self._scenario_path(
            latest_price=latest_price,
            daily_mu=daily_mu,
            daily_volatility=daily_volatility,
            horizon_days=horizon,
            latest_date=latest_date,
            forecast_start_date=forecast_start_date,
            z_score=-scenario_z,
            price_series=price_series,
        )
        base_path = self._scenario_path(
            latest_price=latest_price,
            daily_mu=daily_mu,
            daily_volatility=daily_volatility,
            horizon_days=horizon,
            latest_date=latest_date,
            forecast_start_date=forecast_start_date,
            z_score=0.0,
            price_series=price_series,
        )
        bull_path = self._scenario_path(
            latest_price=latest_price,
            daily_mu=daily_mu,
            daily_volatility=daily_volatility,
            horizon_days=horizon,
            latest_date=latest_date,
            forecast_start_date=forecast_start_date,
            z_score=scenario_z,
            price_series=price_series,
        )
        target_prices = {
            "bear": float(bear_path[-1]["price"]),
            "base": float(base_path[-1]["price"]),
            "bull": float(bull_path[-1]["price"]),
        }
        returns = {
            name: float((target / max(latest_price, 1e-12)) - 1.0)
            for name, target in target_prices.items()
        }
        annualized_volatility = float(daily_volatility * math.sqrt(252))
        max_drawdown = self._historical_drawdown(price_series)
        recent_regimes = np.asarray(asset_context["regimes"][-min(63, len(asset_context["regimes"])) :])
        regime_stability = 1.0
        if recent_regimes.shape[0] > 0:
            counts = np.bincount(recent_regimes.astype(int), minlength=3)
            regime_stability = float(np.max(counts) / recent_regimes.shape[0])
        spread = max(returns["bull"] - returns["bear"], 0.0)
        data_factor = min(float(price_series.shape[0]) / 756.0, 1.0)
        volatility_penalty = min(annualized_volatility / 1.10, 1.0)
        spread_penalty = min(spread / 1.80, 1.0)
        horizon_penalty = min(float(horizon) / 365.0, 1.0) * 0.14
        confidence = float(
            np.clip(
                0.15
                + (0.30 * data_factor)
                + (0.25 * regime_stability)
                + (0.20 * (1.0 - volatility_penalty))
                + (0.10 * (1.0 - spread_penalty)),
                0.05,
                0.95,
            )
        )
        confidence = float(
            np.clip(
                confidence - horizon_penalty,
                0.05,
                0.95,
            )
        )
        risk_label = self._risk_label(annualized_volatility, max_drawdown)
        confidence_label = self._confidence_label(confidence)
        if returns["base"] > 0 and confidence >= 0.45:
            summary = (
                f"{canonical} has a positive base-case scenario with "
                f"{risk_label.lower()} risk and {confidence_label.lower()} confidence."
            )
        elif returns["base"] > 0:
            summary = (
                f"{canonical} has upside in the base case, but the scenario band is wide."
            )
        else:
            summary = (
                f"{canonical} has a weak base-case forecast; treat upside as speculative."
            )

        history_count = min(252, price_series.shape[0])
        historical_prices = [
            {"date": str(date_value), "price": float(price_value)}
            for date_value, price_value in zip(dates[-history_count:], price_series[-history_count:])
        ]
        return {
            "ticker": canonical,
            "asset_class": asset_class,
            "latest_date": str(latest_date),
            "forecast_start_date": forecast_start_date,
            "latest_price": latest_price,
            "horizon_days": horizon,
            "historical_prices": historical_prices,
            "forecast_paths": {
                "bear": bear_path,
                "base": base_path,
                "bull": bull_path,
            },
            "target_prices": target_prices,
            "returns": returns,
            "risk_metrics": {
                "model_estimated_daily_return": daily_mu,
                "annualized_return": float(math.exp(daily_mu * 252) - 1.0),
                "annualized_volatility": annualized_volatility,
                "max_historical_drawdown": max_drawdown,
                "forecast_spread": spread,
                "regime_stability": regime_stability,
            },
            "confidence": confidence,
            "confidence_label": confidence_label,
            "risk_label": risk_label,
            "opportunity_score": self._forecast_score(
                {
                    "returns": returns,
                    "risk_metrics": {"annualized_volatility": annualized_volatility},
                    "confidence": confidence,
                },
                risk=0.5,
            ),
            "return_estimator": {**estimator, **public_estimator},
            "data_as_of": str(latest_date),
            "literacy": {
                "bear_base_bull": (
                    "Bear, base, and bull are scenario ranges, not guaranteed prices."
                ),
                "volatility": "Volatility estimates how much the asset may swing.",
                "drawdown": "Drawdown is the largest historical fall from a prior high.",
                "confidence": "Confidence is lower when data is noisy or scenarios are wide.",
            },
            "plain_language": summary,
        }

    def universe_payload(self) -> dict:
        latest_index = self._combined_context["aligned_rows"] - 1
        latest_date = str(self._combined_context["dates"][latest_index])
        asset_classes = []
        all_tickers = []
        for asset_class, bundle in self.runtime.assets.items():
            prices = self._combined_context["assets"][asset_class]["risky_prices"]
            tickers = []
            for index, ticker in enumerate(bundle.tickers):
                entry = {
                    "ticker": ticker,
                    "asset_class": asset_class,
                    "latest_date": latest_date,
                    "latest_price": float(prices[latest_index, index]),
                    "history_days": int(prices.shape[0]),
                }
                tickers.append(entry)
                all_tickers.append(entry)
            asset_classes.append(
                {
                    "asset_class": asset_class,
                    "tickers": tickers,
                    "history_days": int(prices.shape[0]),
                }
            )
        return {
            "latest_date": latest_date,
            "supported_asset_classes": list(ASSET_CLASSES),
            "asset_classes": asset_classes,
            "tickers": all_tickers,
            "disclaimer": (
                "Forecasts are educational model estimates, not financial advice "
                "or guaranteed returns."
            ),
        }

    def run_ticker_forecast(
        self,
        *,
        ticker: str,
        horizon_days: int,
        window_size: int = 60,
    ) -> dict:
        forecast_start_date = date.today().isoformat()
        return self._ticker_forecast_payload(
            ticker=ticker,
            horizon_days=horizon_days,
            window_size=window_size,
            forecast_start_date=forecast_start_date,
        )

    def _all_ticker_forecasts(
        self,
        *,
        horizon_days: int,
        window_size: int,
    ) -> list[dict]:
        forecast_start_date = date.today().isoformat()
        cache_key = (max(int(horizon_days), 1), max(int(window_size), 2), forecast_start_date)
        if cache_key in self._all_ticker_forecast_cache:
            return [forecast.copy() for forecast in self._all_ticker_forecast_cache[cache_key]]
        forecasts = []
        for bundle in self.runtime.assets.values():
            for ticker in bundle.tickers:
                forecasts.append(
                    self._ticker_forecast_payload(
                        ticker=ticker,
                        horizon_days=horizon_days,
                        window_size=window_size,
                        forecast_start_date=forecast_start_date,
                    )
                )
        self._all_ticker_forecast_cache[cache_key] = [forecast.copy() for forecast in forecasts]
        return forecasts

    def _macro_payload(self) -> dict:
        latest_index = self._combined_context["aligned_rows"] - 1
        macro_values = self._combined_context["global_macro_raw"][latest_index]
        macro_names = self._combined_context["global_macro_feature_names"]
        return {
            "date": str(self._combined_context["dates"][latest_index]),
            "global_regime": int(self._combined_context["combined_regimes"][latest_index]),
            "macro": [
                {"name": name, "value": float(macro_values[index])}
                for index, name in enumerate(macro_names[: len(macro_values)])
            ],
        }

    def run_market_forecast(
        self,
        *,
        horizon_days: int,
        risk: float,
        top_n: int = 10,
        window_size: int = 60,
    ) -> dict:
        forecasts = self._all_ticker_forecasts(
            horizon_days=horizon_days,
            window_size=window_size,
        )
        for forecast in forecasts:
            forecast["opportunity_score"] = self._forecast_score(forecast, risk=risk)
            forecast["risk_score"] = float(
                forecast["risk_metrics"]["annualized_volatility"]
                + max(-forecast["returns"]["bear"], 0.0)
            )
            forecast["risk_adjusted_score"] = float(
                forecast["opportunity_score"] / max(forecast["risk_score"], 0.05)
            )
        ranked = sorted(forecasts, key=lambda entry: entry["opportunity_score"], reverse=True)
        stable = sorted(
            forecasts,
            key=lambda entry: (
                entry["risk_metrics"]["annualized_volatility"],
                -entry["returns"]["base"],
            ),
        )
        downside = sorted(forecasts, key=lambda entry: entry["returns"]["bear"])
        risk_adjusted = sorted(
            forecasts,
            key=lambda entry: entry["risk_adjusted_score"],
            reverse=True,
        )
        best_base = max(forecasts, key=lambda entry: entry["returns"]["base"])
        return {
            "horizon_days": max(int(horizon_days), 1),
            "risk": float(np.clip(risk, 0.0, 1.0)),
            "ranked_tickers": ranked[: max(int(top_n), 1)],
            "highlights": {
                "best_base_case": best_base,
                "best_risk_adjusted": risk_adjusted[0],
                "highest_downside_risk": downside[0],
                "most_stable": stable[0],
            },
            "macro_snapshot": self._macro_payload(),
            "disclaimer": (
                "Rankings are scenario estimates based on historical market data "
                "and should not be treated as trading instructions."
            ),
        }

    def run_portfolio_simulation(
        self,
        *,
        amount: float,
        risk: float,
        horizon_days: int,
        selected_tickers: list[str] | None = None,
        window_size: int = 60,
    ) -> dict:
        risk_value = float(np.clip(risk, 0.0, 1.0))
        if selected_tickers:
            forecast_start_date = date.today().isoformat()
            forecasts = [
                self._ticker_forecast_payload(
                    ticker=ticker,
                    horizon_days=horizon_days,
                    window_size=window_size,
                    forecast_start_date=forecast_start_date,
                )
                for ticker in selected_tickers
            ]
            for forecast in forecasts:
                forecast["opportunity_score"] = self._forecast_score(forecast, risk=risk_value)
            ranked = sorted(forecasts, key=lambda entry: entry["opportunity_score"], reverse=True)
        else:
            ranked = self.run_market_forecast(
                horizon_days=horizon_days,
                risk=risk_value,
                top_n=1000,
                window_size=window_size,
            )["ranked_tickers"]
            ranked = select_diversified_simulation_forecasts(
                ranked,
                risk=risk_value,
                limit=10,
            )

        chosen = ranked[: min(len(ranked), 10)]
        if not chosen:
            raise ArtifactValidationError("No tickers available for portfolio simulation")

        cash_return = float(
            (1.0 + self._cash_daily_return()) ** max(int(horizon_days), 1) - 1.0
        )
        raw_scores = np.asarray(
            [
                max(
                    forecast["opportunity_score"]
                    + (0.25 * max(forecast["returns"]["base"], 0.0))
                    + 0.10,
                    0.01,
                )
                for forecast in chosen
            ],
            dtype=float,
        )
        raw_scores = np.exp(np.clip(raw_scores * 5.0, -5.0, 5.0))
        cash_weight = float(np.clip(0.28 - (0.23 * risk_value), 0.02, 0.35))
        if float(np.mean([forecast["returns"]["base"] for forecast in chosen])) < cash_return:
            cash_weight = min(0.45, cash_weight + 0.12)
        risky_budget = max(1.0 - cash_weight, 0.0)
        max_asset_weight = 0.18 + (0.07 * risk_value)
        risky_weights = allocate_simulation_risky_weights(
            chosen,
            raw_scores,
            risky_budget=risky_budget,
            risk=risk_value,
            max_asset_weight=max_asset_weight,
        )

        allocations = []
        bear_return = cash_weight * cash_return
        base_return = cash_weight * cash_return
        bull_return = cash_weight * cash_return
        confidence_values = []
        for weight, forecast in zip(risky_weights, chosen):
            weight_value = float(weight)
            allocations.append(
                {
                    "ticker": forecast["ticker"],
                    "asset_class": forecast["asset_class"],
                    "weight": weight_value,
                    "amount": float(amount * weight_value),
                    "base_return": float(forecast["returns"]["base"]),
                    "bear_return": float(forecast["returns"]["bear"]),
                    "bull_return": float(forecast["returns"]["bull"]),
                    "confidence": float(forecast["confidence"]),
                }
            )
            bear_return += weight_value * float(forecast["returns"]["bear"])
            base_return += weight_value * float(forecast["returns"]["base"])
            bull_return += weight_value * float(forecast["returns"]["bull"])
            confidence_values.append(float(forecast["confidence"]))

        if cash_weight > 1e-9:
            allocations.append(
                {
                    "ticker": CASH_TICKER,
                    "asset_class": CASH_ASSET_CLASS,
                    "weight": cash_weight,
                    "amount": float(amount * cash_weight),
                    "base_return": cash_return,
                    "bear_return": cash_return,
                    "bull_return": cash_return,
                    "confidence": 0.95,
                }
            )
            confidence_values.append(0.95)

        total_weight = sum(allocation["weight"] for allocation in allocations)
        if total_weight > 1e-12:
            for allocation in allocations:
                allocation["weight"] = float(allocation["weight"] / total_weight)
                allocation["amount"] = float(allocation["weight"] * amount)

        class_weights: dict[str, float] = {}
        for allocation in allocations:
            class_weights[allocation["asset_class"]] = (
                class_weights.get(allocation["asset_class"], 0.0)
                + float(allocation["weight"])
            )
        class_allocations = [
            {
                "asset_class": asset_class,
                "weight": float(weight),
                "amount": float(weight * amount),
            }
            for asset_class, weight in sorted(class_weights.items())
        ]
        trade_plan = [
            {
                "ticker": allocation["ticker"],
                "asset_class": allocation["asset_class"],
                "action": "hold_cash" if allocation["ticker"] == CASH_TICKER else "buy",
                "weight": allocation["weight"],
                "amount": allocation["amount"],
            }
            for allocation in allocations
            if allocation["amount"] >= 1.0
        ]
        warnings = [
            "Forecasts are estimates and should be reviewed before making investment decisions."
        ]
        if max(allocation["weight"] for allocation in allocations) > 0.24:
            warnings.append("One position is above 24%, so concentration risk is elevated.")
        if float(np.mean(confidence_values)) < 0.45:
            warnings.append("Average forecast confidence is low because the scenario band is wide.")
        if cash_weight < 0.03 and risk_value < 0.95:
            warnings.append("Cash is very low for a non-maximum risk setting.")

        return {
            "amount": float(amount),
            "risk": risk_value,
            "horizon_days": max(int(horizon_days), 1),
            "method": "forecast_ranked_scenario_simulator",
            "summary": {
                "bear_value": float(amount * (1.0 + bear_return)),
                "base_value": float(amount * (1.0 + base_return)),
                "bull_value": float(amount * (1.0 + bull_return)),
                "bear_return": float(bear_return),
                "base_return": float(base_return),
                "bull_return": float(bull_return),
                "average_confidence": float(np.mean(confidence_values)),
            },
            "asset_allocations": allocations,
            "class_allocations": class_allocations,
            "trade_plan": trade_plan,
            "source_forecasts": chosen,
            "warnings": warnings,
        }

    def _class_ranges(self) -> dict[str, tuple[int, int]]:
        cursor = 0
        ranges = {}
        for asset_class in ASSET_CLASSES:
            width = len(self.runtime.assets[asset_class].tickers)
            ranges[asset_class] = (cursor, cursor + width)
            cursor += width
        if self.settings.meta_cash_enabled:
            ranges[CASH_ASSET_CLASS] = (cursor, cursor + 1)
        return ranges

    def _asset_names(self) -> list[tuple[str, str]]:
        asset_names = []
        for asset_class in ASSET_CLASSES:
            asset_names.extend(
                [(asset_class, ticker) for ticker in self.runtime.assets[asset_class].tickers]
            )
        if self.settings.meta_cash_enabled:
            asset_names.append((CASH_ASSET_CLASS, CASH_TICKER))
        return asset_names

    def _cash_daily_return(self) -> float:
        return (1.0 + self.settings.meta_cash_annual_return) ** (1.0 / 252.0) - 1.0

    def _risk_cash_cap(self, risk: float) -> float:
        risk_value = float(np.clip(risk, 0.0, 1.0))
        dynamic_cap = 0.08 + ((1.0 - risk_value) * 0.42)
        return float(min(self.settings.meta_max_cash_weight, dynamic_cap))

    def _risk_class_caps(self, risk: float) -> dict[str, float]:
        risk_value = float(np.clip(risk, 0.0, 1.0))
        return {
            "stock": min(self.settings.meta_max_stock_weight, 0.55 + (0.25 * risk_value)),
            "crypto": min(self.settings.meta_max_crypto_weight, 0.04 + (0.31 * risk_value)),
            "etf": min(self.settings.meta_max_etf_weight, 0.72),
            CASH_ASSET_CLASS: self._risk_cash_cap(risk_value),
        }

    def _risk_class_floors(self, risk: float) -> dict[str, float]:
        risk_value = float(np.clip(risk, 0.0, 1.0))
        floors = {
            "stock": 0.12 + (0.08 * risk_value),
            "crypto": 0.02 * risk_value,
            "etf": 0.12 + (0.20 * (1.0 - risk_value)),
            CASH_ASSET_CLASS: 0.03 * (1.0 - risk_value),
        }
        if not self.settings.meta_cash_enabled:
            floors.pop(CASH_ASSET_CLASS, None)
        return floors

    def _diversify_internal_weights(self, weights: np.ndarray, risk: float) -> np.ndarray:
        normalized = normalize_weights(weights)
        equal = np.ones_like(normalized) / max(len(normalized), 1)
        diversity_blend = 0.12 + ((1.0 - float(np.clip(risk, 0.0, 1.0))) * 0.38)
        return normalize_weights(((1.0 - diversity_blend) * normalized) + (diversity_blend * equal))

    def _sub_agent_max_weight(self, bundle: AssetArtifacts) -> float | None:
        config = bundle.metadata.get("ppo_training_config", {})
        max_weight = config.get("max_asset_weight")
        return None if max_weight is None else float(max_weight)

    def _sub_agent_max_cash_weight(self, bundle: AssetArtifacts) -> float | None:
        config = bundle.metadata.get("ppo_training_config", {})
        max_weight = config.get("max_cash_weight")
        return None if max_weight is None else float(max_weight)

    def _sub_agent_signal_blend(self, bundle: AssetArtifacts) -> float:
        return policy_signal_blend_weight(bundle.metadata)

    def _predict_sub_agent_allocation(
        self,
        *,
        bundle: AssetArtifacts,
        asset_context: dict,
        step: int,
        risk: float,
        prev_weights: np.ndarray | None,
        env: SingleAgentEnv | None = None,
    ) -> dict:
        if env is None:
            env = SingleAgentEnv(
                prices=asset_context["prices"],
                ohlcv=asset_context["ohlcv"],
                regimes=asset_context["regimes"],
                micro_indicators=asset_context["micro"],
                macro_indicators=asset_context["macro"],
                risk_appetite=risk,
            )
        observation = env.observation_at(step, prev_weights=prev_weights)
        learned_weights = np.asarray(bundle.policy.predict(observation), dtype=float)
        signal_weights = np.asarray(
            self._single_agent_signal_policies[bundle.asset_class].predict(observation),
            dtype=float,
        )
        blend = self._sub_agent_signal_blend(bundle)
        blended_weights = normalize_weights(
            ((1.0 - blend) * learned_weights) + (blend * signal_weights)
        )
        full_weights = normalize_action_with_cash_sleeve(
            blended_weights,
            risky_asset_count=len(bundle.tickers),
            max_risky_weight=self._sub_agent_max_weight(bundle),
            max_cash_weight=self._sub_agent_max_cash_weight(bundle),
        )
        risky_weights = np.asarray(full_weights[: len(bundle.tickers)], dtype=float)
        cash_weight = max(1.0 - float(risky_weights.sum()), 0.0)
        return {
            "full_weights": np.asarray(full_weights, dtype=float),
            "weights": risky_weights,
            "cash_weight": cash_weight,
            "signal_blend": float(blend),
        }

    def _build_single_agent_envs(self, *, risk: float) -> dict[str, SingleAgentEnv]:
        return {
            asset_class: SingleAgentEnv(
                prices=self._combined_context["assets"][asset_class]["prices"],
                ohlcv=self._combined_context["assets"][asset_class]["ohlcv"],
                regimes=self._combined_context["assets"][asset_class]["regimes"],
                micro_indicators=self._combined_context["assets"][asset_class]["micro"],
                macro_indicators=self._combined_context["assets"][asset_class]["macro"],
                risk_appetite=risk,
            )
            for asset_class in ASSET_CLASSES
        }

    def _compose_meta_sub_agent_signal(self, asset_outputs: dict[str, dict]) -> np.ndarray:
        risky_segments = [asset_outputs[asset_class]["weights"] for asset_class in ASSET_CLASSES]
        if not self._use_v3_meta_architecture():
            signal = normalize_weights(np.concatenate(risky_segments))
            if self.settings.meta_cash_enabled:
                return np.concatenate([signal, np.array([0.0], dtype=float)])
            return signal

        class_count = float(len(ASSET_CLASSES))
        signal = np.concatenate([segment / class_count for segment in risky_segments]).astype(float)
        if not self.settings.meta_cash_enabled:
            return normalize_weights(signal)

        cash_prior = float(
            np.mean([asset_outputs[asset_class]["cash_weight"] for asset_class in ASSET_CLASSES])
        )
        combined = np.concatenate([signal, np.array([cash_prior], dtype=float)])
        total = float(combined.sum())
        if total > 1e-12:
            combined = combined / total
        return combined

    def _meta_max_class_weights(self, risk: float) -> dict[str, float]:
        max_weights = self._risk_class_caps(risk)
        if not self.settings.meta_cash_enabled:
            max_weights.pop(CASH_ASSET_CLASS, None)
        return max_weights

    def _cash_index(self) -> int | None:
        if not self.settings.meta_cash_enabled:
            return None
        return self._class_ranges()[CASH_ASSET_CLASS][0]

    def _asset_weight_caps(self, asset_count: int, risk: float) -> np.ndarray:
        caps = np.full(asset_count, self.settings.meta_max_asset_weight, dtype=float)
        cash_index = self._cash_index()
        if cash_index is not None and cash_index < asset_count:
            caps[cash_index] = self._risk_cash_cap(risk)
        return caps

    def _apply_meta_guardrails(self, weights: np.ndarray, risk: float) -> np.ndarray:
        adjusted = normalize_weights(weights)
        caps = self._asset_weight_caps(len(adjusted), risk)
        for _ in range(3):
            adjusted = apply_class_guardrails(
                adjusted,
                class_ranges=self._class_ranges(),
                max_class_weights=self._meta_max_class_weights(risk),
                max_asset_weight=None,
            )
            adjusted = normalize_weights_with_caps(adjusted, caps)
        return adjusted

    def _apply_risk_adjustments(
        self,
        weights: np.ndarray,
        risk: float,
        *,
        mu_all: np.ndarray | None = None,
        cov_diag: np.ndarray | None = None,
        cash_prior: float | None = None,
    ) -> tuple[np.ndarray, dict | None]:
        adjusted = self._apply_meta_guardrails(weights, risk)
        cash_index = self._cash_index()
        if cash_index is None or mu_all is None or cov_diag is None:
            return adjusted, None

        adjusted, risk_adjustment = apply_cash_risk_managed_overlay(
            adjusted,
            mu_all,
            cov_diag,
            cash_index=cash_index,
            max_cash_weight=self._risk_cash_cap(risk),
            risk_appetite=risk,
            cash_prior=cash_prior,
            target_return=self.settings.meta_min_expected_daily_return,
        )
        adjusted = self._apply_meta_guardrails(adjusted, risk)
        if risk_adjustment is not None:
            risk_adjustment["post_expected_daily_return"] = float(np.dot(adjusted, mu_all))
            risk_adjustment["risk_appetite"] = float(risk)
        return adjusted, risk_adjustment

    def _meta_signal_blend(self) -> float:
        return policy_signal_blend_weight(
            self.runtime.meta.metadata,
            alpha_key="eval_mean_agent_alpha",
            base_blend=0.25,
            max_blend=0.80,
        )

    def _meta_policy_uses_sleeves(self) -> bool:
        return (
            self.runtime.meta.metadata.get("policy_action_layout")
            == "stock_crypto_etf_cash_sleeves"
        )

    def _expand_sleeve_action(
        self,
        sleeve_action: np.ndarray,
        scenario: dict,
        *,
        risk: float,
    ) -> np.ndarray:
        names = list(ASSET_CLASSES)
        if self.settings.meta_cash_enabled:
            names.append(CASH_ASSET_CLASS)
        sleeve_weights = normalize_weights(sleeve_action)
        if sleeve_weights.shape[0] != len(names):
            raise ValueError("Meta sleeve action does not match configured sleeves")

        class_ranges = self._class_ranges()
        weights = np.zeros(len(scenario["mu_all"]), dtype=float)
        for index, asset_class in enumerate(ASSET_CLASSES):
            start, end = class_ranges[asset_class]
            internal_weights = self._diversify_internal_weights(
                scenario["asset_outputs"][asset_class]["weights"],
                risk=risk,
            )
            weights[start:end] = sleeve_weights[index] * internal_weights
        cash_index = self._cash_index()
        if cash_index is not None and CASH_ASSET_CLASS in names:
            weights[cash_index] = sleeve_weights[names.index(CASH_ASSET_CLASS)]
        return normalize_weights(weights)

    def _predict_meta_policy_weights(
        self,
        meta_observation: np.ndarray,
        scenario: dict,
        *,
        risk: float,
    ) -> np.ndarray:
        learned_weights = np.asarray(self.runtime.meta.policy.predict(meta_observation), dtype=float)
        if self._meta_policy_uses_sleeves():
            learned_weights = self._expand_sleeve_action(learned_weights, scenario, risk=risk)
        signal_weights = np.asarray(self._meta_signal_policy.predict(meta_observation), dtype=float)
        blend = self._meta_signal_blend()
        return normalize_weights(
            ((1.0 - blend) * learned_weights) + (blend * signal_weights)
        )

    def _meta_agent_ensemble_blend(self) -> float:
        return policy_signal_blend_weight(
            self.runtime.meta.metadata,
            alpha_key="eval_mean_agent_alpha",
            base_blend=0.30,
            max_blend=0.75,
        )

    def _apply_sleeve_constraints(
        self,
        raw_weights: dict[str, float],
        *,
        risk: float,
    ) -> dict[str, float]:
        classes = list(ASSET_CLASSES)
        if self.settings.meta_cash_enabled:
            classes.append(CASH_ASSET_CLASS)

        floors = self._risk_class_floors(risk)
        caps = self._risk_class_caps(risk)
        floor_total = min(sum(floors.get(name, 0.0) for name in classes), 0.85)
        remaining = max(1.0 - floor_total, 0.0)
        capacity = np.asarray(
            [max(caps.get(name, 1.0) - floors.get(name, 0.0), 0.0) for name in classes],
            dtype=float,
        )
        if remaining <= 1e-12:
            constrained = {name: floors.get(name, 0.0) for name in classes}
            total = sum(constrained.values())
            return {name: value / total for name, value in constrained.items()}

        scores = np.asarray([max(raw_weights.get(name, 0.0), 0.0) for name in classes], dtype=float)
        if float(scores.sum()) <= 1e-12:
            scores = np.ones_like(scores)
        caps_for_remaining = capacity / remaining
        if float(caps_for_remaining.sum()) < 1.0:
            caps_for_remaining = caps_for_remaining / max(float(caps_for_remaining.sum()), 1e-12)
        variable = normalize_weights_with_caps(scores, caps_for_remaining)
        constrained_values = {
            name: floors.get(name, 0.0) + (remaining * float(variable[index]))
            for index, name in enumerate(classes)
        }
        total = sum(constrained_values.values())
        return {name: value / total for name, value in constrained_values.items()}

    def _horizon_diversified_weights(
        self,
        scenario: dict,
        *,
        risk: float,
        duration: int,
    ) -> tuple[np.ndarray, dict]:
        risk_value = float(np.clip(risk, 0.0, 1.0))
        class_ranges = self._class_ranges()
        raw_scores: dict[str, float] = {}
        sleeve_details: dict[str, dict] = {}
        for asset_class in ASSET_CLASSES:
            start, end = class_ranges[asset_class]
            output = scenario["asset_outputs"][asset_class]
            internal_weights = self._diversify_internal_weights(output["weights"], risk_value)
            class_mu = scenario.get("projection_mu_all", scenario["mu_all"])[start:end]
            class_cov = scenario.get("projection_cov_all", scenario["cov_all"])[
                start:end, start:end
            ]
            daily_return = float(np.dot(internal_weights, class_mu))
            daily_volatility = float(np.sqrt(max(internal_weights.T @ class_cov @ internal_weights, 0.0)))
            horizon_return = float(max((1.0 + daily_return) ** max(int(duration), 1) - 1.0, -0.95))
            horizon_volatility = float(daily_volatility * math.sqrt(max(int(duration), 1)))
            concentration = float(np.sum(internal_weights**2))
            risk_penalty = (0.20 + ((1.0 - risk_value) * 0.90)) * horizon_volatility
            diversification_penalty = (0.05 + ((1.0 - risk_value) * 0.25)) * concentration
            score = horizon_return - risk_penalty - diversification_penalty
            raw_scores[asset_class] = float(np.exp(np.clip(score * 4.0, -6.0, 6.0)))
            sleeve_details[asset_class] = {
                "expected_daily_return": daily_return,
                "projected_horizon_return": horizon_return,
                "horizon_volatility": horizon_volatility,
                "concentration": concentration,
                "internal_weights": internal_weights,
            }

        if self.settings.meta_cash_enabled:
            cash_return = float((1.0 + self._cash_daily_return()) ** max(int(duration), 1) - 1.0)
            risky_scores = [sleeve_details[name]["projected_horizon_return"] for name in ASSET_CLASSES]
            weak_market_pressure = max(-float(np.mean(risky_scores)), 0.0)
            cash_score = cash_return + ((1.0 - risk_value) * 0.04) + (weak_market_pressure * 0.35)
            raw_scores[CASH_ASSET_CLASS] = float(np.exp(np.clip(cash_score * 4.0, -6.0, 6.0)))
            sleeve_details[CASH_ASSET_CLASS] = {
                "expected_daily_return": self._cash_daily_return(),
                "projected_horizon_return": cash_return,
                "horizon_volatility": 0.0,
                "concentration": 1.0,
            }

        sleeve_weights = self._apply_sleeve_constraints(raw_scores, risk=risk_value)
        weights = np.zeros(len(scenario["mu_all"]), dtype=float)
        for asset_class in ASSET_CLASSES:
            start, end = class_ranges[asset_class]
            internal_weights = sleeve_details[asset_class]["internal_weights"]
            weights[start:end] = sleeve_weights[asset_class] * internal_weights
        cash_index = self._cash_index()
        if cash_index is not None and CASH_ASSET_CLASS in sleeve_weights:
            weights[cash_index] = sleeve_weights[CASH_ASSET_CLASS]
        return self._apply_meta_guardrails(weights, risk_value), {
            "sleeve_weights": {name: float(value) for name, value in sleeve_weights.items()},
            "sleeve_scores": {name: float(value) for name, value in raw_scores.items()},
            "sleeve_details": {
                name: {
                    key: float(value)
                    for key, value in detail.items()
                    if key != "internal_weights"
                }
                for name, detail in sleeve_details.items()
            },
            "risk_cash_cap": self._risk_cash_cap(risk_value),
        }

    def _predict_mixed_policy_weights(
        self,
        scenario: dict,
        *,
        risk: float,
        duration: int,
    ) -> tuple[np.ndarray, dict]:
        meta_weights = self._predict_meta_policy_weights(
            scenario["meta_observation"],
            scenario,
            risk=risk,
        )
        agent_weights = np.asarray(scenario["sub_agent_weights"], dtype=float)
        horizon_weights, horizon_details = self._horizon_diversified_weights(
            scenario,
            risk=risk,
            duration=duration,
        )
        agent_blend = self._meta_agent_ensemble_blend()
        meta_agent_mix = blend_allocation_sources(
            meta_weights,
            agent_weights,
            secondary_weight=agent_blend,
        )
        risk_value = float(np.clip(risk, 0.0, 1.0))
        horizon_blend = 0.60 + (risk_value * 0.25)
        mixed_weights = blend_allocation_sources(
            meta_agent_mix,
            horizon_weights,
            secondary_weight=horizon_blend,
        )
        return mixed_weights, {
            "meta_signal_blend": self._meta_signal_blend(),
            "sub_agent_consensus_blend": float(agent_blend),
            "horizon_diversification_blend": float(horizon_blend),
            "return_estimator": scenario.get("return_estimator", {}),
            **horizon_details,
        }

    def _max_meta_turnover(self, risk: float) -> float:
        risk_value = float(np.clip(risk, 0.0, 1.0))
        return 0.16 + (0.08 * risk_value)

    def _build_class_features(
        self,
        *,
        asset_outputs: dict[str, dict],
        mu_all: np.ndarray,
        cov_all: np.ndarray,
        prev_weights: np.ndarray,
    ) -> np.ndarray:
        features = []
        class_ranges = self._class_ranges()
        for asset_class in ASSET_CLASSES:
            start, end = class_ranges[asset_class]
            class_signal = np.asarray(asset_outputs[asset_class]["weights"], dtype=float)
            if float(class_signal.sum()) > 1e-12:
                class_weights = class_signal / float(class_signal.sum())
            else:
                class_weights = np.ones(end - start, dtype=float) / max(end - start, 1)
            class_mu = mu_all[start:end]
            class_cov = cov_all[start:end, start:end]
            class_expected_return = float(np.dot(class_weights, class_mu))
            class_volatility = float(np.sqrt(max(class_weights.T @ class_cov @ class_weights, 0.0)))
            class_prev_weight = float(prev_weights[start:end].sum())
            features.extend([class_expected_return, class_volatility, class_prev_weight])
            if self._use_v3_meta_architecture():
                features.append(float(asset_outputs[asset_class].get("cash_weight", 0.0)))
        return np.asarray(features, dtype=np.float32)

    def _build_snapshot(
        self,
        step: int,
        sub_agent_allocations: dict[str, np.ndarray],
        *,
        policy_mix: dict | None = None,
        risk_adjustment: dict | None = None,
    ) -> dict:
        stock_macro = self._combined_context["global_macro_raw"][step]
        macro_names = self._combined_context["global_macro_feature_names"]
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
            "policy_mix": policy_mix
            or {
                "meta_signal_blend": self._meta_signal_blend(),
                "sub_agent_consensus_blend": self._meta_agent_ensemble_blend(),
            },
            "risk_adjustment": risk_adjustment or {"applied": False},
        }

    def _build_step_scenario(
        self,
        *,
        step: int,
        risk: float,
        window_size: int,
        prev_weights: np.ndarray | None,
        prev_sub_weights: dict[str, np.ndarray] | None = None,
        single_agent_envs: dict[str, SingleAgentEnv] | None = None,
    ) -> dict:
        asset_outputs = {}
        mu_blocks = []
        cov_blocks = []
        projection_mu_blocks = []
        projection_cov_blocks = []
        return_estimator = {}

        for asset_class in ASSET_CLASSES:
            bundle = self.runtime.assets[asset_class]
            asset_context = self._combined_context["assets"][asset_class]
            sub_prev_weights = prev_sub_weights.get(asset_class) if prev_sub_weights else None
            output = self._predict_sub_agent_allocation(
                bundle=bundle,
                asset_context=asset_context,
                step=step,
                risk=risk,
                prev_weights=sub_prev_weights,
                env=single_agent_envs.get(asset_class) if single_agent_envs else None,
            )
            mu, cov = self._compute_mu_cov(asset_context["risky_prices"], window_size, step)
            projection_mu, projection_cov, projection_details = self._compute_projection_mu_cov(
                asset_context["risky_prices"],
                window_size,
                step,
                asset_class=asset_class,
            )
            asset_outputs[asset_class] = {
                **output,
                "mu": mu,
                "cov": cov,
                "projection_mu": projection_mu,
                "projection_cov": projection_cov,
            }
            mu_blocks.append(mu)
            cov_blocks.append(cov)
            projection_mu_blocks.append(projection_mu)
            projection_cov_blocks.append(projection_cov)
            return_estimator[asset_class] = projection_details

        mu_all = np.concatenate(mu_blocks)
        cov_all = _block_diagonal(cov_blocks)
        projection_mu_all = np.concatenate(projection_mu_blocks)
        projection_cov_all = _block_diagonal(projection_cov_blocks)
        sub_agent_weights = self._compose_meta_sub_agent_signal(asset_outputs)
        if self.settings.meta_cash_enabled:
            mu_all = np.concatenate(
                [mu_all, np.array([self._cash_daily_return()], dtype=float)]
            )
            cash_cov = np.zeros((cov_all.shape[0] + 1, cov_all.shape[1] + 1), dtype=float)
            cash_cov[:-1, :-1] = cov_all
            cash_cov[-1, -1] = 1e-10
            cov_all = cash_cov
            projection_mu_all = np.concatenate(
                [projection_mu_all, np.array([self._cash_daily_return()], dtype=float)]
            )
            projection_cash_cov = np.zeros(
                (projection_cov_all.shape[0] + 1, projection_cov_all.shape[1] + 1),
                dtype=float,
            )
            projection_cash_cov[:-1, :-1] = projection_cov_all
            projection_cash_cov[-1, -1] = 1e-10
            projection_cov_all = projection_cash_cov
            return_estimator[CASH_ASSET_CLASS] = {
                "method": "fixed_cash_rate",
                "expected_daily_return": self._cash_daily_return(),
            }
        base_prev_weights = (
            sub_agent_weights
            if prev_weights is None
            else normalize_weights(prev_weights)
        )
        class_features = self._build_class_features(
            asset_outputs=asset_outputs,
            mu_all=mu_all,
            cov_all=cov_all,
            prev_weights=base_prev_weights,
        )

        meta_env = MetaPortfolioEnv(
            micro_indicators=self._combined_context["combined_micro"],
            macro_indicators=self._combined_context["combined_macro"],
            regimes=self._combined_context["combined_regimes"],
            n_assets=len(mu_all),
            class_feature_dim=self._meta_class_feature_dim(),
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
            "projection_mu_all": projection_mu_all,
            "projection_cov_all": projection_cov_all,
            "return_estimator": return_estimator,
            "sub_agent_weights": sub_agent_weights,
            "class_features": class_features,
            "meta_observation": meta_observation,
            "prev_weights": base_prev_weights,
        }

    def _projection_summary(
        self,
        *,
        amount: float,
        weights: np.ndarray,
        mu_all: np.ndarray,
        cov_all: np.ndarray,
        duration: int,
    ) -> dict[str, float]:
        horizon_days = max(int(duration), 1)
        expected_daily_return = float(np.dot(weights, mu_all))
        if abs(expected_daily_return) < 1e-12:
            expected_daily_return = 0.0
        portfolio_variance = float(weights.T @ cov_all @ weights)
        daily_volatility = float(math.sqrt(max(portfolio_variance, 0.0)))
        annualized_return = float(max((1.0 + expected_daily_return) ** 252 - 1.0, -1.0))
        annualized_volatility = float(daily_volatility * math.sqrt(252))
        projected_horizon_return = float(
            max((1.0 + expected_daily_return) ** horizon_days - 1.0, -1.0)
        )
        projected_value = float(amount * (1.0 + projected_horizon_return))
        horizon_volatility = daily_volatility * math.sqrt(horizon_days)
        downside_return = float(max(projected_horizon_return - horizon_volatility, -1.0))
        upside_return = float(projected_horizon_return + horizon_volatility)
        return {
            "expected_daily_return": expected_daily_return,
            "annualized_return": annualized_return,
            "portfolio_variance": portfolio_variance,
            "annualized_volatility": annualized_volatility,
            "projected_horizon_return": projected_horizon_return,
            "projected_value": projected_value,
            "projected_profit": float(projected_value - amount),
            "downside_value": float(amount * (1.0 + downside_return)),
            "upside_value": float(amount * (1.0 + upside_return)),
        }

    def _asset_prices_at_step(self, step: int) -> np.ndarray:
        prices = []
        for asset_class in ASSET_CLASSES:
            prices.extend(self._combined_context["assets"][asset_class]["risky_prices"][step])
        if self.settings.meta_cash_enabled:
            prices.append(1.0)
        return np.asarray(prices, dtype=float)

    def _build_initial_trade_log(
        self,
        *,
        step: int,
        amount: float,
        allocations: list[dict],
    ) -> list[dict]:
        prices = self._asset_prices_at_step(step)
        trade_log = []
        for index, allocation in enumerate(allocations):
            allocation_amount = float(allocation["amount"])
            if allocation_amount <= 1e-9:
                continue
            price = 1.0 if allocation["asset_class"] == CASH_ASSET_CLASS else float(prices[index])
            shares = allocation_amount / max(price, 1e-12)
            trade_log.append(
                {
                    "date": str(self._combined_context["dates"][step]),
                    "agent": allocation["asset_class"],
                    "ticker": allocation["ticker"],
                    "action": "buy" if allocation["asset_class"] != CASH_ASSET_CLASS else "hold_cash",
                    "weight_before": 0.0,
                    "weight_after": float(allocation["weight"]),
                    "amount": allocation_amount,
                    "price": price,
                    "shares": float(shares),
                    "portfolio_value_before": float(amount),
                    "portfolio_value_after": float(amount),
                    "transaction_cost": 0.0,
                    "reason": "initial horizon/diversification allocation",
                }
            )
        return trade_log

    def _build_rebalance_trade_log(
        self,
        *,
        step: int,
        portfolio_value: float,
        previous_weights: np.ndarray,
        target_weights: np.ndarray,
        transaction_fee: float,
    ) -> list[dict]:
        prices = self._asset_prices_at_step(step)
        asset_names = self._asset_names()
        trade_log = []
        deltas = np.asarray(target_weights, dtype=float) - np.asarray(previous_weights, dtype=float)
        for index, delta in enumerate(deltas):
            if abs(float(delta)) <= 5e-4:
                continue
            asset_class, ticker = asset_names[index]
            trade_amount = float(delta * portfolio_value)
            price = 1.0 if asset_class == CASH_ASSET_CLASS else float(prices[index])
            trade_log.append(
                {
                    "date": str(self._combined_context["dates"][step]),
                    "agent": asset_class,
                    "ticker": ticker,
                    "action": "buy" if trade_amount > 0 else "sell",
                    "weight_before": float(previous_weights[index]),
                    "weight_after": float(target_weights[index]),
                    "amount": abs(trade_amount),
                    "price": price,
                    "shares": float(abs(trade_amount) / max(price, 1e-12)),
                    "portfolio_value_before": float(portfolio_value),
                    "portfolio_value_after": float(portfolio_value),
                    "transaction_cost": float(abs(delta) * transaction_fee * portfolio_value),
                    "reason": "horizon/diversification rebalance",
                }
            )
        return trade_log

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
        raw_weights, policy_mix = self._predict_mixed_policy_weights(
            scenario,
            risk=risk,
            duration=duration,
        )
        final_weights, risk_adjustment = self._apply_risk_adjustments(
            raw_weights,
            risk,
            mu_all=scenario["projection_mu_all"],
            cov_diag=np.diag(scenario["projection_cov_all"]),
            cash_prior=(
                None
                if self._cash_index() is None
                else float(scenario["sub_agent_weights"][self._cash_index()])
            ),
        )
        asset_names = self._asset_names()

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

        projection = self._projection_summary(
            amount=amount,
            weights=final_weights,
            mu_all=scenario["projection_mu_all"],
            cov_all=scenario["projection_cov_all"],
            duration=duration,
        )

        top_asset_targets = [
            allocation["ticker"]
            for allocation in sorted(
                (
                    allocation
                    for allocation in asset_allocations
                    if allocation["asset_class"] != CASH_ASSET_CLASS
                ),
                key=lambda entry: entry["weight"],
                reverse=True,
            )[: self.settings.top_asset_target_count]
        ]

        feature_slices = build_feature_slices(
            n_assets=len(asset_names),
            micro_dim=self._combined_context["combined_micro"].shape[1],
            macro_dim=self._combined_context["combined_macro"].shape[1],
            class_feature_dim=self._meta_class_feature_dim(),
        )

        warnings = list(self._combined_context["warnings"])
        if risk_adjustment is not None:
            warnings.append(
                "Cash exposure was adjusted inside the risk-based cash cap, not allowed "
                "to dominate the portfolio."
            )

        return InferenceResult(
            summary=InferenceSummary(
                expected_daily_return=projection["expected_daily_return"],
                annualized_return=projection["annualized_return"],
                portfolio_variance=projection["portfolio_variance"],
                annualized_volatility=projection["annualized_volatility"],
                projected_horizon_return=projection["projected_horizon_return"],
                projected_value=projection["projected_value"],
                projected_profit=projection["projected_profit"],
                downside_value=projection["downside_value"],
                upside_value=projection["upside_value"],
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
                policy_mix=policy_mix,
                risk_adjustment=risk_adjustment,
            ),
            warnings=warnings,
            model_version=str(self.runtime.meta.metadata.get("feature_version", "unknown")),
            top_asset_targets=top_asset_targets,
            feature_slices=feature_slices,
            latest_observation=scenario["meta_observation"],
            trade_log=self._build_initial_trade_log(
                step=step,
                amount=amount,
                allocations=asset_allocations,
            ),
        )

    def run_backtest(
        self,
        *,
        initial_amount: float,
        risk: float,
        window_size: int,
        max_steps: int,
        include_trade_log: bool = False,
    ) -> BacktestResult:
        prices_all = self._combined_context["combined_prices"]
        if self.settings.meta_cash_enabled:
            cash_prices = np.cumprod(
                np.full(prices_all.shape[0], 1.0 + self._cash_daily_return(), dtype=float)
            ).reshape(-1, 1)
            prices_all = np.hstack([prices_all, cash_prices])
        max_available_steps = prices_all.shape[0] - window_size - 1
        steps_to_run = min(max_available_steps, max_steps)
        if steps_to_run <= 0:
            raise ArtifactValidationError("Not enough aligned rows to run a backtest")

        transaction_fee = float(
            self.runtime.meta.metadata.get("sac_training_config", {}).get("transaction_fee", 0.0)
        )
        portfolio_value = float(initial_amount)
        prev_weights = None
        prev_sub_weights = None
        start_step = prices_all.shape[0] - steps_to_run - 1
        start_step = max(start_step, window_size)
        equity_curve = [
            {
                "step": 0,
                "date": str(self._combined_context["dates"][start_step]),
                "value": portfolio_value,
            }
        ]
        drawdown_curve = [
            {
                "step": 0,
                "date": str(self._combined_context["dates"][start_step]),
                "value": 0.0,
            }
        ]
        returns = []
        peak = portfolio_value
        risk_off_applied_steps = 0
        turnover_history: list[float] = []
        transaction_cost_history: list[float] = []
        trade_log: list[dict] = []
        single_agent_envs = self._build_single_agent_envs(risk=risk)

        for offset, step in enumerate(range(start_step, start_step + steps_to_run), start=1):
            scenario = self._build_step_scenario(
                step=step,
                risk=risk,
                window_size=window_size,
                prev_weights=prev_weights,
                prev_sub_weights=prev_sub_weights,
                single_agent_envs=single_agent_envs,
            )
            raw_weights, _ = self._predict_mixed_policy_weights(
                scenario,
                risk=risk,
                duration=max_steps,
            )
            weights, risk_adjustment = self._apply_risk_adjustments(
                raw_weights,
                risk,
                mu_all=scenario["mu_all"],
                cov_diag=np.diag(scenario["cov_all"]),
                cash_prior=(
                    None
                    if self._cash_index() is None
                    else float(scenario["sub_agent_weights"][self._cash_index()])
                ),
            )
            if risk_adjustment is not None:
                risk_off_applied_steps += 1
            if prev_weights is None:
                cash_index = self._cash_index()
                if cash_index is None:
                    previous_weights = scenario["prev_weights"]
                else:
                    previous_weights = np.zeros_like(weights, dtype=float)
                    previous_weights[cash_index] = 1.0
            else:
                previous_weights = prev_weights
            weights = constrain_turnover(
                weights,
                previous_weights,
                max_turnover=self._max_meta_turnover(risk),
            )
            trade_previous_weights = previous_weights
            turnover = float(np.sum(np.abs(weights - previous_weights)))
            transaction_cost = turnover * transaction_fee
            if include_trade_log:
                trade_log.extend(
                    self._build_rebalance_trade_log(
                        step=step,
                        portfolio_value=portfolio_value,
                        previous_weights=trade_previous_weights,
                        target_weights=weights,
                        transaction_fee=transaction_fee,
                    )
                )
            actual_returns = (
                prices_all[step + 1] - prices_all[step]
            ) / np.clip(prices_all[step], 1e-12, None)
            gross_return = float(np.dot(weights, actual_returns))
            daily_return = gross_return - transaction_cost
            portfolio_value *= max(1 + daily_return, 1e-6)
            peak = max(peak, portfolio_value)
            drawdown = 1 - (portfolio_value / peak)
            point_date = str(self._combined_context["dates"][step + 1])
            equity_curve.append(
                {"step": offset, "date": point_date, "value": float(portfolio_value)}
            )
            drawdown_curve.append(
                {"step": offset, "date": point_date, "value": float(drawdown)}
            )
            returns.append(daily_return)
            turnover_history.append(turnover)
            transaction_cost_history.append(transaction_cost)
            prev_weights = weights
            prev_sub_weights = {
                asset_class: scenario["asset_outputs"][asset_class]["full_weights"]
                for asset_class in ASSET_CLASSES
            }

        cumulative_return = float((portfolio_value / initial_amount) - 1)
        mean_daily_return = float(np.mean(returns))
        daily_volatility = float(np.std(returns))
        annualized_return = float(
            max((portfolio_value / initial_amount) ** (252.0 / steps_to_run) - 1.0, -1.0)
        )
        annualized_volatility = float(daily_volatility * math.sqrt(252))
        sharpe_ratio = (
            float((mean_daily_return / daily_volatility) * math.sqrt(252))
            if daily_volatility > 0
            else 0.0
        )
        max_drawdown = float(max(point["value"] for point in drawdown_curve))

        warnings = list(self._combined_context["warnings"])
        if risk_off_applied_steps:
            warnings.append(
                f"Risk-capped cash adjustment applied on {risk_off_applied_steps} "
                f"of {steps_to_run} backtest steps."
            )
        train_rows = int(self.runtime.meta.metadata.get("train_rows", 0))
        if train_rows > 0 and start_step >= (train_rows - 1):
            warnings.append(
                "Backtest is fully inside the meta-model validation period used "
                "during model selection; treat it as validation, not final holdout."
            )
        elif train_rows > 0 and (train_rows - 1) < (start_step + steps_to_run):
            warnings.append(
                "Backtest overlaps the meta-model validation period used during "
                "model selection; treat it as validation, not final holdout."
            )
        warnings.append(
            f"Backtest evaluated on the most recent {steps_to_run} aligned trading days."
        )
        if steps_to_run < 252:
            warnings.append(
                "Backtest horizon is shorter than 252 trading days, so annualized "
                "metrics may be unstable."
            )

        return BacktestResult(
            summary_metrics={
                "cumulative_return": cumulative_return,
                "annualized_return": annualized_return,
                "daily_volatility": daily_volatility,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "ending_value": float(portfolio_value),
                "backtest_steps": float(steps_to_run),
                "average_daily_turnover": float(np.mean(turnover_history) if turnover_history else 0.0),
                "average_daily_transaction_cost": float(
                    np.mean(transaction_cost_history) if transaction_cost_history else 0.0
                ),
            },
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            trade_log=trade_log,
            warnings=warnings,
        )


@lru_cache(maxsize=4)
def get_engine(settings: Settings, *, strict_validation: bool = True) -> ForesightEngine:
    return ForesightEngine(settings, strict_validation=strict_validation)


def reset_engine() -> None:
    get_engine.cache_clear()
