from __future__ import annotations

import numpy as np
import pandas as pd

from offline.market_pipeline import ASSET_UNIVERSES, MACRO_FEATURE_COLUMNS, OHLCV_FIELDS
from offline.rebuild_market_data import refresh_asset_class


def test_rebuild_market_data_smoke_with_synthetic_frame(tmp_path):
    asset_class = "etf"
    tickers = ASSET_UNIVERSES[asset_class]["tickers"]
    dates = pd.date_range("2025-01-01", periods=130, freq="D")
    columns = pd.MultiIndex.from_product([tickers, OHLCV_FIELDS])
    frame = pd.DataFrame(index=dates, columns=columns, dtype=float)

    for ticker_index, ticker in enumerate(tickers):
        base = 50.0 + ticker_index
        trend = np.linspace(0.0, 8.0, len(dates))
        close = base + trend + np.sin(np.arange(len(dates)) / 7.0)
        frame[(ticker, "Open")] = close * 0.995
        frame[(ticker, "High")] = close * 1.01
        frame[(ticker, "Low")] = close * 0.99
        frame[(ticker, "Close")] = close
        frame[(ticker, "Volume")] = 1_000_000 + (ticker_index * 10_000)

    macro = pd.DataFrame({"Date": dates})
    for column_index, column in enumerate(MACRO_FEATURE_COLUMNS):
        macro[column] = 1.0 + column_index + np.linspace(0.0, 0.1, len(dates))
    macro_path = tmp_path / "macro.csv"
    macro.to_csv(macro_path, index=False)

    report = refresh_asset_class(
        asset_class=asset_class,
        artifact_root=tmp_path / "processed",
        dataset_root=tmp_path / "datasets",
        macro_path=macro_path,
        end_date="2025-06-01",
        fit_new_scalers=True,
        skip_scaler_write=False,
        market_frame=frame,
    )

    assert report["rows"] == 130
    assert report["uses_ohlcv_features"] is True
    assert (tmp_path / "processed" / asset_class / "prices.npy").exists()
    assert (tmp_path / "processed" / asset_class / "metadata.json").exists()
