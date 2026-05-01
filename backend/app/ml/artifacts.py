from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np

from backend.app.ml.envs import single_agent_observation_dim
from backend.app.ml.errors import ArtifactValidationError
from backend.app.ml.numpy_compat import install_numpy_pickle_compat
from backend.app.ml.policies import load_policy


ASSET_CLASSES = ("stock", "crypto", "etf")
DEFAULT_MACRO_NAMES = [
    "vix_market_volatility",
    "federal_funds_rate",
    "ten_year_treasury_yield",
    "unemployment_rate",
    "cpi_all_items",
    "recession_indicator",
]


@dataclass
class AlignmentReport:
    original_rows: dict[str, int]
    aligned_rows: int
    trimmed: bool


@dataclass
class AssetArtifacts:
    asset_class: str
    tickers: list[str]
    dates: np.ndarray
    prices: np.ndarray
    ohlcv: np.ndarray
    regimes: np.ndarray
    micro_indicators: np.ndarray
    macro_indicators: np.ndarray
    macro_indicators_raw: np.ndarray
    indicator_scaler: object | None
    macro_scaler: object | None
    policy: object
    metadata: dict
    alignment: AlignmentReport


@dataclass
class MetaArtifacts:
    policy: object
    metadata: dict
    model_path: Path
    macro_scaler: object | None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return _read_json(path)


def _load_optional_pickle(path: Path):
    if not path.exists():
        return None
    install_numpy_pickle_compat()
    return joblib.load(path)


def _synthetic_ohlcv_from_prices(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    ohlcv = np.zeros((prices.shape[0], prices.shape[1], 5), dtype=float)
    ohlcv[:, :, 0] = prices
    ohlcv[:, :, 1] = prices
    ohlcv[:, :, 2] = prices
    ohlcv[:, :, 3] = prices
    ohlcv[:, :, 4] = 1.0
    return ohlcv


def _append_cash_sleeve_to_market_data(
    prices: np.ndarray,
    ohlcv: np.ndarray,
    *,
    cash_annual_return: float,
) -> tuple[np.ndarray, np.ndarray]:
    daily_return = (1.0 + float(cash_annual_return)) ** (1.0 / 252.0) - 1.0
    cash_prices = np.cumprod(
        np.full(np.asarray(prices).shape[0], 1.0 + daily_return, dtype=float)
    ).reshape(-1, 1)
    cash_ohlcv = np.zeros((cash_prices.shape[0], 1, 5), dtype=float)
    cash_ohlcv[:, 0, 0] = cash_prices[:, 0]
    cash_ohlcv[:, 0, 1] = cash_prices[:, 0]
    cash_ohlcv[:, 0, 2] = cash_prices[:, 0]
    cash_ohlcv[:, 0, 3] = cash_prices[:, 0]
    cash_ohlcv[:, 0, 4] = 1.0
    return np.hstack([np.asarray(prices, dtype=float), cash_prices]), np.concatenate(
        [np.asarray(ohlcv, dtype=float), cash_ohlcv],
        axis=1,
    )


def _default_dates(row_count: int) -> np.ndarray:
    return np.arange(int(row_count))


def _ensure_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ArtifactValidationError(f"{name} contains NaN or infinite values")


def _load_asset_policy(
    model_path: Path,
    metadata: dict,
    *,
    base_observation_dim: int,
    action_dim: int,
    legacy_observation_dim: int | None = None,
    previous_observation_dims: tuple[int, ...] = (),
):
    candidate_dims = [base_observation_dim]
    expanded_dim = base_observation_dim + action_dim
    if expanded_dim not in candidate_dims:
        candidate_dims.append(expanded_dim)
    if legacy_observation_dim is not None:
        for candidate in (legacy_observation_dim, legacy_observation_dim + action_dim):
            if candidate not in candidate_dims:
                candidate_dims.append(candidate)
    for candidate in previous_observation_dims:
        for candidate_dim in (candidate, candidate + action_dim):
            if candidate_dim not in candidate_dims:
                candidate_dims.append(candidate_dim)

    last_error: Exception | None = None
    for candidate_dim in candidate_dims:
        try:
            policy = load_policy(
                model_path,
                metadata,
                observation_dim=candidate_dim,
                action_dim=action_dim,
            )
            return policy, candidate_dim
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ArtifactValidationError(f"Unable to load policy at {model_path}")


def _default_asset_metadata(asset_class: str) -> dict:
    return {
        "asset_class": asset_class,
        "feature_version": "legacy-rl-v1",
        "policy_backend": "sb3",
        "algorithm": "ppo",
        "model_file": "model.zip",
        "macro_feature_names": DEFAULT_MACRO_NAMES,
    }


def _default_meta_metadata() -> dict:
    return {
        "feature_version": "legacy-rl-v1",
        "policy_backend": "sb3",
        "algorithm": "sac",
        "model_file": "model.zip",
    }


def validate_and_align_asset_artifacts(
    *,
    tickers: list[str],
    dates: np.ndarray | None,
    prices: np.ndarray,
    ohlcv: np.ndarray | None,
    regimes: np.ndarray,
    micro_indicators: np.ndarray,
    macro_indicators: np.ndarray,
    strict: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    AlignmentReport,
]:
    original_rows = {
        "dates": int(dates.shape[0]) if dates is not None else int(prices.shape[0]),
        "prices": int(prices.shape[0]),
        "regimes": int(regimes.shape[0]),
        "micro_indicators": int(micro_indicators.shape[0]),
        "macro_indicators": int(macro_indicators.shape[0]),
    }
    if ohlcv is not None:
        original_rows["ohlcv"] = int(ohlcv.shape[0])
    unique_rows = set(original_rows.values())
    if strict and len(unique_rows) != 1:
        raise ArtifactValidationError(f"Mismatched artifact rows: {original_rows}")

    aligned_rows = min(unique_rows)
    if aligned_rows < 2:
        raise ArtifactValidationError(
            f"Aligned artifact length is too small for inference: {aligned_rows}"
        )

    dates = (
        np.asarray(dates)[-aligned_rows:]
        if dates is not None
        else _default_dates(aligned_rows)
    )
    prices = np.asarray(prices, dtype=float)[-aligned_rows:]
    if ohlcv is None:
        ohlcv = _synthetic_ohlcv_from_prices(prices)
    else:
        ohlcv = np.asarray(ohlcv, dtype=float)[-aligned_rows:]
    regimes = np.asarray(regimes, dtype=int).reshape(-1)[-aligned_rows:]
    micro_indicators = np.asarray(micro_indicators, dtype=float)[-aligned_rows:]
    macro_indicators = np.asarray(macro_indicators, dtype=float)[-aligned_rows:]

    if prices.ndim != 2:
        raise ArtifactValidationError("prices must be a 2D array")
    if ohlcv.ndim != 3 or ohlcv.shape[2] != 5:
        raise ArtifactValidationError("ohlcv must be a 3D array with shape (rows, assets, 5)")
    if micro_indicators.ndim != 2 or macro_indicators.ndim != 2:
        raise ArtifactValidationError("indicator arrays must be 2D")
    if prices.shape[1] != len(tickers):
        raise ArtifactValidationError(
            f"Ticker count {len(tickers)} does not match prices columns {prices.shape[1]}"
        )
    if ohlcv.shape[1] != len(tickers):
        raise ArtifactValidationError(
            f"Ticker count {len(tickers)} does not match ohlcv asset dimension {ohlcv.shape[1]}"
        )
    if dates.ndim != 1:
        raise ArtifactValidationError("dates must be a 1D array")
    if np.unique(dates).shape[0] != dates.shape[0]:
        raise ArtifactValidationError("dates must not contain duplicates")

    _ensure_finite("prices", prices)
    _ensure_finite("ohlcv", ohlcv)
    _ensure_finite("micro_indicators", micro_indicators)
    _ensure_finite("macro_indicators", macro_indicators)

    return (
        dates,
        prices,
        ohlcv,
        regimes,
        micro_indicators,
        macro_indicators,
        AlignmentReport(
            original_rows=original_rows,
            aligned_rows=aligned_rows,
            trimmed=len(unique_rows) != 1,
        ),
    )


def load_asset_artifacts(root: Path, asset_class: str, *, strict: bool) -> AssetArtifacts:
    asset_dir = root / asset_class
    if not asset_dir.exists():
        raise ArtifactValidationError(f"Missing asset directory: {asset_dir}")

    metadata_path = asset_dir / "metadata.json"
    metadata = _default_asset_metadata(asset_class)
    if metadata_path.exists():
        metadata.update(_read_json(metadata_path))

    tickers = json.loads((asset_dir / "tickers.json").read_text())
    dates_path = asset_dir / "dates.npy"
    dates = np.load(dates_path) if dates_path.exists() else None
    prices = np.load(asset_dir / "prices.npy")
    ohlcv_path = asset_dir / "ohlcv.npy"
    ohlcv = np.load(ohlcv_path) if ohlcv_path.exists() else None
    regimes = np.load(asset_dir / "regimes.npy")
    micro_indicators = np.load(asset_dir / "micro_indicators.npy")
    macro_indicators = np.load(asset_dir / "macro_indicators.npy")
    macro_indicators_raw_path = asset_dir / "macro_indicators_raw.npy"
    macro_indicators_raw = (
        np.load(macro_indicators_raw_path)
        if macro_indicators_raw_path.exists()
        else np.asarray(macro_indicators, dtype=float)
    )

    (
        dates,
        prices,
        ohlcv,
        regimes,
        micro_indicators,
        macro_indicators,
        alignment,
    ) = validate_and_align_asset_artifacts(
        tickers=tickers,
        dates=dates,
        prices=prices,
        ohlcv=ohlcv,
        regimes=regimes,
        micro_indicators=micro_indicators,
        macro_indicators=macro_indicators,
        strict=strict,
    )
    if strict and int(macro_indicators_raw.shape[0]) != alignment.aligned_rows:
        raise ArtifactValidationError(
            "Mismatched artifact rows: "
            f"{{'macro_indicators_raw': {int(macro_indicators_raw.shape[0])}, "
            f"'aligned_rows': {alignment.aligned_rows}}}"
        )

    feature_names = _read_optional_json(asset_dir / "feature_names.json")
    raw_macro_feature_names = feature_names.get(
        "macro_raw",
        feature_names.get("macro", metadata.get("macro_feature_names", DEFAULT_MACRO_NAMES)),
    )
    training_config = metadata.get("ppo_training_config", {})
    action_dim = int(metadata.get("action_dim", len(tickers)))
    cash_enabled = bool(training_config.get("cash_enabled")) or action_dim > len(tickers)
    if cash_enabled and action_dim <= len(tickers):
        action_dim = len(tickers) + 1
    if cash_enabled:
        prices, ohlcv = _append_cash_sleeve_to_market_data(
            prices,
            ohlcv,
            cash_annual_return=float(training_config.get("cash_annual_return", 0.04)),
        )

    inference_observation_dim = single_agent_observation_dim(
        n_assets=action_dim,
        micro_dim=int(micro_indicators.shape[1]),
        macro_dim=int(macro_indicators.shape[1]),
    )
    legacy_observation_dim = (
        action_dim + 2 + 3 + int(micro_indicators.shape[1]) + int(macro_indicators.shape[1])
    )
    ohlcv_v2_observation_dim = (
        (action_dim * 8)
        + 2
        + 3
        + int(micro_indicators.shape[1])
        + int(macro_indicators.shape[1])
    )
    model_file = metadata.get("model_file", "model.zip")
    policy, policy_observation_dim = _load_asset_policy(
        asset_dir / model_file,
        metadata,
        base_observation_dim=inference_observation_dim,
        action_dim=action_dim,
        legacy_observation_dim=legacy_observation_dim,
        previous_observation_dims=(ohlcv_v2_observation_dim,),
    )
    metadata["aligned_rows"] = alignment.aligned_rows
    metadata["ticker_count"] = len(tickers)
    metadata["micro_feature_count"] = int(micro_indicators.shape[1])
    metadata["macro_feature_count"] = int(macro_indicators.shape[1])
    metadata["raw_macro_feature_names"] = raw_macro_feature_names
    metadata["cash_enabled"] = cash_enabled
    metadata["inference_observation_dim"] = inference_observation_dim
    metadata["policy_observation_dim"] = policy_observation_dim
    metadata["action_dim"] = action_dim

    return AssetArtifacts(
        asset_class=asset_class,
        tickers=tickers,
        dates=dates,
        prices=prices,
        ohlcv=ohlcv,
        regimes=regimes,
        micro_indicators=micro_indicators,
        macro_indicators=macro_indicators,
        macro_indicators_raw=macro_indicators_raw[-alignment.aligned_rows:],
        indicator_scaler=_load_optional_pickle(asset_dir / "indicator_scaler.pkl"),
        macro_scaler=_load_optional_pickle(asset_dir / "macro_scaler.pkl"),
        policy=policy,
        metadata=metadata,
        alignment=alignment,
    )


def load_meta_artifacts(
    root: Path,
    *,
    observation_dim: int | None = None,
    action_dim: int | None = None,
) -> MetaArtifacts:
    meta_dir = root / "meta"
    if not meta_dir.exists():
        raise ArtifactValidationError(f"Missing meta directory: {meta_dir}")

    metadata = _default_meta_metadata()
    metadata_path = meta_dir / "metadata.json"
    if metadata_path.exists():
        metadata.update(_read_json(metadata_path))

    model_path = meta_dir / metadata.get("model_file", "model.zip")
    candidate_dims: list[int | None] = [observation_dim]
    for candidate in (
        metadata.get("policy_observation_dim"),
        metadata.get("inference_observation_dim"),
        observation_dim + 1 if observation_dim is not None else None,
        observation_dim - 1 if observation_dim is not None and observation_dim > 1 else None,
    ):
        if candidate is not None and candidate not in candidate_dims:
            candidate_dims.append(int(candidate))
    if None not in candidate_dims:
        candidate_dims.append(None)

    last_error: Exception | None = None
    policy = None
    for candidate_dim in candidate_dims:
        try:
            policy = load_policy(
                model_path,
                metadata,
                observation_dim=candidate_dim,
                action_dim=action_dim,
            )
            metadata["policy_observation_dim"] = policy.observation_dim
            if observation_dim is not None:
                metadata["inference_observation_dim"] = observation_dim
            break
        except Exception as exc:
            last_error = exc

    if policy is None:
        if last_error is not None:
            raise last_error
        raise ArtifactValidationError(f"Unable to load meta policy at {model_path}")
    return MetaArtifacts(
        policy=policy,
        metadata=metadata,
        model_path=model_path,
        macro_scaler=_load_optional_pickle(meta_dir / "meta_macro_scaler.pkl"),
    )


def peek_meta_metadata(root: Path) -> dict:
    meta_dir = root / "meta"
    metadata = _default_meta_metadata()
    metadata_path = meta_dir / "metadata.json"
    if metadata_path.exists():
        metadata.update(_read_json(metadata_path))
    return metadata
