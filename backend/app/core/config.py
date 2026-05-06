from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    project_name: str
    api_prefix: str
    artifact_root: Path
    dataset_root: Path
    surrogate_sample_size: int
    surrogate_fidelity_threshold: float
    top_asset_target_count: int
    default_backtest_steps: int
    meta_max_asset_weight: float
    meta_max_stock_weight: float
    meta_max_crypto_weight: float
    meta_max_etf_weight: float
    meta_max_cash_weight: float
    meta_min_expected_daily_return: float
    meta_cash_enabled: bool
    meta_cash_annual_return: float
    market_data_provider: str
    market_index_auto_refresh: bool
    market_index_config_path: Path
    market_index_refresh_lookback_days: int
    supabase_url: str
    supabase_service_role_key: str
    require_supabase: bool
    load_artifact_engine: bool
    lazy_load_artifact_engine: bool
    artifact_policy_mode: str


_settings: Settings | None = None


def _env(name: str, default: object = "") -> str:
    preferred = os.getenv(f"FORESIGHT_{name}")
    if preferred is not None:
        return preferred

    legacy = os.getenv(f"STOCKIFY_{name}")
    if legacy is not None:
        return legacy

    return str(default)


def _env_bool(name: str, default: str) -> bool:
    return _env(name, default).lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    global _settings
    if _settings is not None:
        return _settings

    _settings = Settings(
        project_name="Foresight Backend",
        api_prefix="/api",
        artifact_root=Path(_env("ARTIFACT_ROOT", REPO_ROOT / "artifacts" / "processed")),
        dataset_root=Path(_env("DATASET_ROOT", REPO_ROOT / "datasets" / "raw")),
        surrogate_sample_size=int(_env("SURROGATE_SAMPLE_SIZE", "128")),
        surrogate_fidelity_threshold=float(_env("SURROGATE_FIDELITY_THRESHOLD", "0.55")),
        top_asset_target_count=int(_env("TOP_ASSET_TARGET_COUNT", "3")),
        default_backtest_steps=int(_env("DEFAULT_BACKTEST_STEPS", "252")),
        meta_max_asset_weight=float(_env("META_MAX_ASSET_WEIGHT", "0.20")),
        meta_max_stock_weight=float(_env("META_MAX_STOCK_WEIGHT", "0.85")),
        meta_max_crypto_weight=float(_env("META_MAX_CRYPTO_WEIGHT", "0.30")),
        meta_max_etf_weight=float(_env("META_MAX_ETF_WEIGHT", "0.70")),
        meta_max_cash_weight=float(_env("META_MAX_CASH_WEIGHT", "0.95")),
        meta_min_expected_daily_return=float(_env("META_MIN_EXPECTED_DAILY_RETURN", "0.0")),
        meta_cash_enabled=_env_bool("META_CASH_ENABLED", "true"),
        meta_cash_annual_return=float(_env("META_CASH_ANNUAL_RETURN", "0.04")),
        market_data_provider=_env("MARKET_DATA_PROVIDER", "yfinance"),
        market_index_auto_refresh=_env_bool("MARKET_INDEX_AUTO_REFRESH", "false"),
        market_index_config_path=Path(
            _env(
                "MARKET_INDEX_CONFIG_PATH",
                REPO_ROOT / "config" / "market_indices.v1.json",
            )
        ),
        market_index_refresh_lookback_days=int(
            _env("MARKET_INDEX_REFRESH_LOOKBACK_DAYS", "10")
        ),
        supabase_url=os.getenv("SUPABASE_URL", ""),
        supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
        require_supabase=_env_bool("REQUIRE_SUPABASE", "false"),
        load_artifact_engine=_env_bool("LOAD_ARTIFACT_ENGINE", "true"),
        lazy_load_artifact_engine=_env_bool("LAZY_LOAD_ARTIFACT_ENGINE", "false"),
        artifact_policy_mode=_env("ARTIFACT_POLICY_MODE", "trained").strip().lower(),
    )
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None
