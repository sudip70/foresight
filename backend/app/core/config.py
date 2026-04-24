from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        project_name="Stockify Backend",
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
    )
