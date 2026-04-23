from pathlib import Path
import json

import numpy as np
import pytest

from backend.app.ml.artifacts import ArtifactValidationError, validate_and_align_asset_artifacts
from backend.tests.helpers import build_fixture_artifact_tree


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
    stock_dir = artifact_root / "stock"
    metadata = json.loads((stock_dir / "metadata.json").read_text())
    prices = np.load(stock_dir / "prices.npy")
    assert metadata["feature_version"] == "fixture-v1"
    assert prices.shape == (24, 2)
