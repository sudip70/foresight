import json

import numpy as np
import pytest

from backend.app.ml.artifacts import ArtifactValidationError, validate_and_align_asset_artifacts
from backend.app.ml.envs import meta_observation_dim
from backend.tests.helpers import build_fixture_artifact_tree
from offline.repair_processed_artifacts import repair_asset


def test_validate_and_align_fails_fast_in_strict_mode():
    tickers = ["AAPL", "MSFT"]
    dates = np.arange(10)
    prices = np.ones((10, 2))
    regimes = np.zeros(9, dtype=int)
    micro = np.ones((10, 4))
    macro = np.ones((10, 2))

    with pytest.raises(ArtifactValidationError):
        validate_and_align_asset_artifacts(
            tickers=tickers,
            dates=dates,
            prices=prices,
            ohlcv=None,
            regimes=regimes,
            micro_indicators=micro,
            macro_indicators=macro,
            strict=True,
        )


def test_fixture_artifacts_are_written_in_expected_shape(tmp_path):
    artifact_root = build_fixture_artifact_tree(tmp_path)
    for asset_class in ("stock", "crypto", "etf"):
        asset_dir = artifact_root / asset_class
        metadata = json.loads((asset_dir / "metadata.json").read_text())
        prices = np.load(asset_dir / "prices.npy")
        dates = np.load(asset_dir / "dates.npy")
        regimes = np.load(asset_dir / "regimes.npy")
        micro = np.load(asset_dir / "micro_indicators.npy")
        macro = np.load(asset_dir / "macro_indicators.npy")

        assert metadata["feature_version"] == "fixture-v1"
        assert prices.shape == (24, 2)
        assert dates.shape == (24,)
        assert regimes.shape == (24,)
        assert micro.shape == (24, 4)
        assert macro.shape == (24, 2)

    meta_model = json.loads((artifact_root / "meta" / "model.json").read_text())
    assert meta_model["observation_dim"] == meta_observation_dim(
        n_assets=7,
        micro_dim=12,
        macro_dim=6,
    )


def test_repair_processed_artifacts_dry_run_and_backup(tmp_path):
    artifact_root = build_fixture_artifact_tree(tmp_path)
    stock_dir = artifact_root / "stock"
    prices_path = stock_dir / "prices.npy"
    prices = np.load(prices_path)
    np.save(prices_path, prices[:-2])

    dry_run = repair_asset("stock", artifact_root=artifact_root, dry_run=True)
    assert dry_run["trimmed"] is True
    assert np.load(prices_path).shape[0] == 22
    assert not list(stock_dir.glob("*.backup_before_repair_*.npy"))

    repaired = repair_asset("stock", artifact_root=artifact_root, dry_run=False)
    row_counts = {
        "dates": np.load(stock_dir / "dates.npy").shape[0],
        "prices": np.load(stock_dir / "prices.npy").shape[0],
        "ohlcv": np.load(stock_dir / "ohlcv.npy").shape[0],
        "regimes": np.load(stock_dir / "regimes.npy").shape[0],
        "micro": np.load(stock_dir / "micro_indicators.npy").shape[0],
        "macro": np.load(stock_dir / "macro_indicators.npy").shape[0],
    }
    assert set(row_counts.values()) == {22}
    assert repaired["array_backups"]["prices.npy"].endswith(".npy")
    assert list(stock_dir.glob("prices.backup_before_repair_*.npy"))
