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


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is not None:
        return _settings

    _settings = Settings(
        project_name="Foresight Backend",
        api_prefix="/api",
        artifact_root=Path(
            os.getenv("STOCKIFY_ARTIFACT_ROOT", REPO_ROOT / "artifacts" / "processed")
        ),
        dataset_root=Path(
            os.getenv("STOCKIFY_DATASET_ROOT", REPO_ROOT / "datasets" / "raw")
        ),
        surrogate_sample_size=int(os.getenv("STOCKIFY_SURROGATE_SAMPLE_SIZE", "128")),
        surrogate_fidelity_threshold=float(
            os.getenv("STOCKIFY_SURROGATE_FIDELITY_THRESHOLD", "0.55")
        ),
        top_asset_target_count=int(os.getenv("STOCKIFY_TOP_ASSET_TARGET_COUNT", "3")),
        default_backtest_steps=int(os.getenv("STOCKIFY_DEFAULT_BACKTEST_STEPS", "252")),
        meta_max_asset_weight=float(os.getenv("STOCKIFY_META_MAX_ASSET_WEIGHT", "0.20")),
        meta_max_stock_weight=float(os.getenv("STOCKIFY_META_MAX_STOCK_WEIGHT", "0.85")),
        meta_max_crypto_weight=float(os.getenv("STOCKIFY_META_MAX_CRYPTO_WEIGHT", "0.30")),
        meta_max_etf_weight=float(os.getenv("STOCKIFY_META_MAX_ETF_WEIGHT", "0.70")),
        meta_max_cash_weight=float(os.getenv("STOCKIFY_META_MAX_CASH_WEIGHT", "0.95")),
        meta_min_expected_daily_return=float(
            os.getenv("STOCKIFY_META_MIN_EXPECTED_DAILY_RETURN", "0.0")
        ),
        meta_cash_enabled=os.getenv("STOCKIFY_META_CASH_ENABLED", "true").lower()
        in {"1", "true", "yes", "on"},
        meta_cash_annual_return=float(os.getenv("STOCKIFY_META_CASH_ANNUAL_RETURN", "0.04")),
        market_data_provider=os.getenv("STOCKIFY_MARKET_DATA_PROVIDER", "yfinance"),
        market_index_auto_refresh=os.getenv(
            "STOCKIFY_MARKET_INDEX_AUTO_REFRESH", "false"
        ).lower()
        in {"1", "true", "yes", "on"},
        market_index_config_path=Path(
            os.getenv(
                "STOCKIFY_MARKET_INDEX_CONFIG_PATH",
                REPO_ROOT / "config" / "market_indices.v1.json",
            )
        ),
        market_index_refresh_lookback_days=int(
            os.getenv("STOCKIFY_MARKET_INDEX_REFRESH_LOOKBACK_DAYS", "10")
        ),
        supabase_url=os.getenv("SUPABASE_URL", ""),
        supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    )
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None
