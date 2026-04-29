from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from offline.market_pipeline import (
    ASSET_UNIVERSES,
    MACRO_FEATURE_COLUMNS,
    OHLCV_FIELDS,
    align_macro_frame,
    build_asset_dataset,
    classify_regimes,
    compute_micro_indicator_frame,
    extract_asset_frames,
)


def test_asset_universes_use_expanded_config() -> None:
    assert len(ASSET_UNIVERSES["stock"]["tickers"]) == 25
    assert len(ASSET_UNIVERSES["etf"]["tickers"]) == 20
    assert "AMD" in ASSET_UNIVERSES["stock"]["tickers"]
    assert "XLK" in ASSET_UNIVERSES["etf"]["tickers"]


def _market_frame_for(tickers: list[str], periods: int = 140) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    payload = {}
    trend = np.linspace(0.0, 20.0, periods)
    seasonal = np.sin(np.linspace(0.0, 8.0, periods))

    for offset, ticker in enumerate(tickers):
        base = 100 + (offset * 7) + trend + seasonal
        open_ = base * (1 + 0.002 * np.cos(np.linspace(0.0, 6.0, periods)))
        close = base * (1 + 0.001 * np.sin(np.linspace(0.0, 5.0, periods)))
        high = np.maximum(open_, close) * 1.01
        low = np.minimum(open_, close) * 0.99
        volume = 1_000_000 + (offset * 10_000) + np.arange(periods) * 500
        for field, values in {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }.items():
            payload[(ticker, field)] = values

    return pd.DataFrame(payload, index=index)


def _macro_csv(path: Path, dates: pd.DatetimeIndex) -> Path:
    macro_dates = pd.date_range(dates.min(), periods=8, freq="MS")
    frame = pd.DataFrame(
        {
            "Date": macro_dates,
            "VIX Market Volatility": np.linspace(14.0, 21.0, len(macro_dates)),
            "Federal Funds Rate": np.linspace(4.0, 4.5, len(macro_dates)),
            "10-Year Treasury Yield": np.linspace(3.8, 4.2, len(macro_dates)),
            "Unemployment Rate": np.linspace(3.7, 4.0, len(macro_dates)),
            "CPI All Items": np.linspace(305.0, 312.0, len(macro_dates)),
            "Recession Indicator": np.zeros(len(macro_dates)),
        }
    )
    frame.to_csv(path, index=False)
    return path


def test_compute_micro_indicator_frame_uses_ohlcv_inputs() -> None:
    frame = _market_frame_for(["AAA", "BBB"])
    asset_frames = extract_asset_frames(frame, ["AAA", "BBB"])

    features = compute_micro_indicator_frame(asset_frames)

    assert features.shape == (140, 20)
    assert not features.isna().any().any()
    assert "AAA__atr" in features.columns
    assert "BBB__obv" in features.columns


def test_align_macro_frame_forward_fills_to_market_dates(tmp_path: Path) -> None:
    market_dates = pd.date_range("2024-01-01", periods=50, freq="B")
    macro_path = _macro_csv(tmp_path / "macro.csv", market_dates)

    aligned = align_macro_frame(macro_path, market_dates)

    assert list(aligned.columns) == list(MACRO_FEATURE_COLUMNS)
    assert aligned.index.equals(market_dates)
    assert not aligned.isna().any().any()


def test_classify_regimes_returns_three_state_encoding() -> None:
    index = pd.date_range("2024-01-01", periods=120, freq="B")
    close = pd.Series(np.linspace(100.0, 140.0, len(index)), index=index)

    regimes = classify_regimes(close)

    assert len(regimes) == len(close)
    assert set(regimes.unique()).issubset({0, 1, 2})


def test_build_asset_dataset_writes_ohlcv_and_feature_artifacts(tmp_path: Path) -> None:
    tickers = list(ASSET_UNIVERSES["etf"]["tickers"])
    market_frame = _market_frame_for(tickers)
    macro_path = _macro_csv(tmp_path / "macro.csv", market_frame.index)

    dataset = build_asset_dataset(
        asset_class="etf",
        market_frame=market_frame,
        macro_path=macro_path,
        artifact_root=tmp_path / "artifacts",
        raw_output_path=tmp_path / "raw" / "etf_ohlcv.csv",
        reuse_existing_scalers=False,
        persist_scalers=True,
    )

    assert dataset.prices.shape == (140, len(tickers))
    assert dataset.ohlcv.shape == (140, len(tickers), len(OHLCV_FIELDS))
    assert dataset.micro_indicators_scaled.shape[1] == len(tickers) * 10
    assert dataset.macro_indicators_scaled.shape[1] == len(MACRO_FEATURE_COLUMNS)
    assert dataset.metadata["feature_version"] == "ohlcv-v2"
    assert dataset.metadata["uses_ohlcv_features"] is True

    asset_dir = tmp_path / "artifacts" / "etf"
    assert (asset_dir / "prices.npy").exists()
    assert (asset_dir / "ohlcv.npy").exists()
    assert (asset_dir / "micro_indicators_raw.npy").exists()
    assert (asset_dir / "micro_indicators.npy").exists()
    assert (asset_dir / "macro_indicators_raw.npy").exists()
    assert (asset_dir / "macro_indicators.npy").exists()
    assert (asset_dir / "dates.npy").exists()
    assert (asset_dir / "feature_names.json").exists()
    assert (tmp_path / "raw" / "etf_ohlcv.csv").exists()


def test_build_asset_dataset_replaces_stale_policy_metadata(tmp_path: Path) -> None:
    tickers = list(ASSET_UNIVERSES["stock"]["tickers"])
    benchmark = str(ASSET_UNIVERSES["stock"]["benchmark"])
    frame_tickers = tickers if benchmark in tickers else [*tickers, benchmark]
    market_frame = _market_frame_for(frame_tickers)
    macro_path = _macro_csv(tmp_path / "macro.csv", market_frame.index)
    asset_dir = tmp_path / "artifacts" / "stock"
    asset_dir.mkdir(parents=True)
    (asset_dir / "metadata.json").write_text(
        json.dumps({"policy_backend": "sb3", "model_file": "model.zip", "action_dim": 11})
    )

    build_asset_dataset(
        asset_class="stock",
        market_frame=market_frame,
        macro_path=macro_path,
        artifact_root=tmp_path / "artifacts",
        reuse_existing_scalers=False,
        persist_scalers=True,
    )

    metadata = json.loads((asset_dir / "metadata.json").read_text())
    assert metadata["policy_backend"] == "single_agent_signal"
    assert metadata["action_dim"] == len(tickers) + 1
    assert metadata["previous_action_dim"] == 11
