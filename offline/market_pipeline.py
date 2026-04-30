from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
import json
import os
import shutil
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume


OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Volume")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE_CONFIG_PATH = REPO_ROOT / "config" / "asset_universe.v1.json"
DEFAULT_CLASS_BENCHMARKS = {
    "stock": "SPY",
    "crypto": "BTC-USD",
    "etf": "SPY",
}
MACRO_FEATURE_COLUMNS = (
    "VIX Market Volatility",
    "Federal Funds Rate",
    "10-Year Treasury Yield",
    "Unemployment Rate",
    "CPI All Items",
    "Recession Indicator",
)
MICRO_FEATURES = (
    "rsi",
    "macd_diff",
    "bb_width",
    "stoch_k",
    "obv",
    "adx",
    "psar",
    "ichimoku_diff",
    "williams_r",
    "atr",
)

def load_asset_universes(path: Path = DEFAULT_UNIVERSE_CONFIG_PATH) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text())
    assets = payload.get("assets", [])
    if not isinstance(assets, list) or not assets:
        raise ValueError(f"No assets found in universe config: {path}")

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in assets:
        if not row.get("active", True):
            continue
        asset_class = str(row["asset_class"])
        grouped.setdefault(asset_class, []).append(row)

    universes: dict[str, dict[str, object]] = {}
    for asset_class, rows in sorted(grouped.items()):
        tickers = [str(row.get("provider_symbol") or row["ticker"]).strip().upper() for row in rows]
        start_dates = [
            str(row.get("start_date") or payload.get("default_start_date") or "2018-01-01")
            for row in rows
        ]
        universes[asset_class] = {
            "tickers": tickers,
            "benchmark": DEFAULT_CLASS_BENCHMARKS.get(asset_class, tickers[0]),
            "start_date": min(start_dates),
        }
    return universes


ASSET_UNIVERSES = load_asset_universes()


@dataclass(frozen=True)
class AssetDataset:
    asset_class: str
    tickers: list[str]
    dates: pd.DatetimeIndex
    prices: np.ndarray
    ohlcv: np.ndarray
    regimes: np.ndarray
    micro_indicators_raw: np.ndarray
    micro_indicators_scaled: np.ndarray
    macro_indicators_raw: np.ndarray
    macro_indicators_scaled: np.ndarray
    metadata: dict


def default_end_date() -> str:
    return (date.today() + timedelta(days=1)).isoformat()


def _ensure_multiindex_download(frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        if len(tickers) != 1:
            raise ValueError("Expected a multi-index market frame for multiple tickers")
        frame = pd.concat({tickers[0]: frame}, axis=1)

    if frame.columns.nlevels != 2:
        raise ValueError("Market frame must use a 2-level multi-index")

    level0 = list(frame.columns.get_level_values(0))
    level1 = list(frame.columns.get_level_values(1))
    if set(level0).issubset(set(OHLCV_FIELDS)) and set(level1).issubset(set(tickers)):
        frame = frame.swaplevel(axis=1)

    frame = frame.sort_index(axis=1)
    return frame


def _normalize_asset_frame(frame: pd.DataFrame) -> pd.DataFrame:
    asset_frame = frame.loc[:, [field for field in OHLCV_FIELDS if field in frame.columns]].copy()
    missing = [field for field in OHLCV_FIELDS if field not in asset_frame.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    asset_frame = asset_frame.sort_index()
    asset_frame = asset_frame.replace([np.inf, -np.inf], np.nan)
    asset_frame["Volume"] = asset_frame["Volume"].fillna(0.0)
    asset_frame[list(OHLCV_FIELDS[:-1])] = asset_frame[list(OHLCV_FIELDS[:-1])].ffill()
    asset_frame = asset_frame.dropna(subset=["Open", "High", "Low", "Close"])
    asset_frame["High"] = asset_frame[["Open", "High", "Low", "Close"]].max(axis=1)
    asset_frame["Low"] = asset_frame[["Open", "High", "Low", "Close"]].min(axis=1)
    if (asset_frame["Close"] <= 0).any():
        raise ValueError("Close prices must be positive for indicator generation")
    return asset_frame.astype(float)


def flatten_ohlcv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    flattened = frame.copy()
    flattened.columns = [f"{ticker}__{field}" for ticker, field in flattened.columns]
    return flattened


def download_ohlcv_history(
    tickers: list[str],
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    import yfinance as yf

    frame = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if frame.empty:
        raise ValueError(f"No market data returned for tickers: {tickers}")
    return _ensure_multiindex_download(frame, tickers)


def extract_asset_frames(frame: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    normalized = _ensure_multiindex_download(frame, tickers)
    asset_frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        if ticker not in normalized.columns.get_level_values(0):
            raise ValueError(f"Ticker {ticker} not found in downloaded market frame")
        asset_frames[ticker] = _normalize_asset_frame(normalized[ticker])
    return asset_frames


def compute_micro_indicator_frame(asset_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    feature_frames = []
    for ticker, frame in asset_frames.items():
        close = frame["Close"]
        high = frame["High"]
        low = frame["Low"]
        volume = frame["Volume"]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module=r"ta\.trend",
            )
            psar = ta.trend.PSARIndicator(high=high, low=low, close=close).psar()

        ticker_features = pd.DataFrame(
            {
                f"{ticker}__rsi": ta.momentum.RSIIndicator(close=close).rsi().fillna(50.0),
                f"{ticker}__macd_diff": ta.trend.MACD(close=close).macd_diff().fillna(0.0),
                f"{ticker}__bb_width": ta.volatility.BollingerBands(
                    close=close
                ).bollinger_wband().fillna(0.0),
                f"{ticker}__stoch_k": ta.momentum.StochasticOscillator(
                    high=high, low=low, close=close, window=14, smooth_window=3
                ).stoch().fillna(0.0),
                f"{ticker}__obv": ta.volume.OnBalanceVolumeIndicator(
                    close=close, volume=volume
                ).on_balance_volume().fillna(0.0),
                f"{ticker}__adx": ta.trend.ADXIndicator(
                    high=high, low=low, close=close, window=14
                ).adx().fillna(0.0),
                f"{ticker}__psar": psar.bfill().ffill().fillna(close),
                f"{ticker}__ichimoku_diff": (
                    ta.trend.IchimokuIndicator(high=high, low=low)
                    .ichimoku_conversion_line()
                    .bfill()
                    - ta.trend.IchimokuIndicator(high=high, low=low)
                    .ichimoku_base_line()
                    .bfill()
                ).fillna(0.0),
                f"{ticker}__williams_r": ta.momentum.WilliamsRIndicator(
                    high=high, low=low, close=close
                ).williams_r().fillna(-50.0),
                f"{ticker}__atr": ta.volatility.AverageTrueRange(
                    high=high, low=low, close=close
                ).average_true_range().bfill().fillna(0.0),
            },
            index=frame.index,
        )
        feature_frames.append(ticker_features)

    combined = pd.concat(feature_frames, axis=1).sort_index(axis=1)
    combined = combined.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return combined


def align_macro_frame(macro_path: Path, dates: pd.DatetimeIndex) -> pd.DataFrame:
    macro = pd.read_csv(macro_path, parse_dates=["Date"])
    required = ["Date", *MACRO_FEATURE_COLUMNS]
    missing = [column for column in required if column not in macro.columns]
    if missing:
        raise ValueError(f"Macro dataset is missing required columns: {missing}")

    macro = macro.loc[:, required].copy()
    macro = macro.sort_values("Date").set_index("Date")
    macro = macro.replace([np.inf, -np.inf], np.nan).ffill()
    macro = macro.reindex(dates).ffill().bfill()
    if macro.isna().any().any():
        raise ValueError("Macro dataset could not be aligned to market dates")
    return macro.astype(float)


def classify_regimes(benchmark_close: pd.Series) -> pd.Series:
    daily_returns = benchmark_close.pct_change().fillna(0.0)
    rolling_return = benchmark_close.pct_change(63).fillna(0.0)
    rolling_volatility = daily_returns.rolling(21).std().bfill().fillna(0.0)
    low_vol = float(rolling_volatility.quantile(0.33))
    high_vol = float(rolling_volatility.quantile(0.66))

    bullish = (rolling_return > 0) & (rolling_volatility <= high_vol)
    bearish = (rolling_return < 0) & (rolling_volatility >= low_vol)

    regimes = pd.Series(1, index=benchmark_close.index, dtype=int)
    regimes.loc[bullish] = 0
    regimes.loc[bearish] = 2
    return regimes


def _scale_frame(
    frame: pd.DataFrame,
    *,
    scaler_path: Path,
    reuse_existing: bool,
    persist_scaler: bool,
) -> tuple[np.ndarray, MinMaxScaler]:
    if reuse_existing and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        try:
            scaled = scaler.transform(frame)
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
            return scaled.astype(float), scaler
        except Exception:
            scaler = MinMaxScaler()
    else:
        scaler = MinMaxScaler()

    scaler.fit(frame)
    if persist_scaler:
        joblib.dump(scaler, scaler_path)

    scaled = scaler.transform(frame)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled.astype(float), scaler


def _stage_artifact_dir(asset_dir: Path) -> Path:
    asset_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{asset_dir.name}.staging.",
            dir=str(asset_dir.parent),
        )
    )
    if asset_dir.exists():
        shutil.copytree(asset_dir, staging_dir, dirs_exist_ok=True)
    return staging_dir


def _commit_staged_artifact_dir(staging_dir: Path, asset_dir: Path) -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = asset_dir.with_name(f".{asset_dir.name}.backup_before_commit_{timestamp}")
    if asset_dir.exists():
        os.rename(asset_dir, backup_dir)
    try:
        os.rename(staging_dir, asset_dir)
    except Exception:
        if backup_dir.exists():
            os.rename(backup_dir, asset_dir)
        raise


def save_raw_market_csv(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flatten_ohlcv_columns(frame).to_csv(output_path, index=True, index_label="Date")
    return output_path


def build_asset_dataset(
    *,
    asset_class: str,
    market_frame: pd.DataFrame,
    macro_path: Path,
    artifact_root: Path,
    raw_output_path: Path | None = None,
    reuse_existing_scalers: bool = True,
    persist_scalers: bool = True,
) -> AssetDataset:
    if asset_class not in ASSET_UNIVERSES:
        raise ValueError(f"Unknown asset class: {asset_class}")

    config = ASSET_UNIVERSES[asset_class]
    tickers = list(config["tickers"])
    benchmark = config["benchmark"]
    tickers_for_frame = list(tickers)
    if benchmark not in tickers_for_frame:
        tickers_for_frame.append(benchmark)

    normalized_frame = _ensure_multiindex_download(market_frame, tickers_for_frame)
    asset_frames = extract_asset_frames(normalized_frame, tickers)
    benchmark_frame = extract_asset_frames(normalized_frame, [benchmark])[benchmark]

    shared_dates = benchmark_frame.index
    for frame in asset_frames.values():
        shared_dates = shared_dates.intersection(frame.index)
    shared_dates = shared_dates.sort_values()
    if len(shared_dates) < 100:
        raise ValueError(
            f"Not enough aligned market rows for {asset_class}: {len(shared_dates)}"
        )

    asset_frames = {ticker: frame.loc[shared_dates] for ticker, frame in asset_frames.items()}
    benchmark_frame = benchmark_frame.loc[shared_dates]
    micro_frame = compute_micro_indicator_frame(asset_frames).loc[shared_dates]
    macro_frame = align_macro_frame(macro_path, shared_dates)
    regime_series = classify_regimes(benchmark_frame["Close"]).loc[shared_dates]

    prices = np.column_stack([asset_frames[ticker]["Close"].values for ticker in tickers]).astype(float)
    ohlcv = np.stack(
        [asset_frames[ticker].loc[:, OHLCV_FIELDS].values for ticker in tickers],
        axis=1,
    ).astype(float)

    asset_dir = artifact_root / asset_class
    staging_asset_dir = _stage_artifact_dir(asset_dir)
    metadata_path = staging_asset_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    existing_backend = str(metadata.get("policy_backend", "sb3")).lower()
    existing_macro_names = metadata.get("macro_feature_names")
    if existing_backend == "sb3" and isinstance(existing_macro_names, list):
        compatible_macro_names = [
            str(name) for name in existing_macro_names if str(name) in macro_frame.columns
        ]
        if compatible_macro_names:
            macro_frame = macro_frame.loc[:, compatible_macro_names]

    micro_scaled, _ = _scale_frame(
        micro_frame,
        scaler_path=staging_asset_dir / "indicator_scaler.pkl",
        reuse_existing=reuse_existing_scalers,
        persist_scaler=persist_scalers,
    )
    macro_scaled, _ = _scale_frame(
        macro_frame,
        scaler_path=staging_asset_dir / "macro_scaler.pkl",
        reuse_existing=reuse_existing_scalers,
        persist_scaler=persist_scalers,
    )

    np.save(staging_asset_dir / "prices.npy", prices)
    np.save(staging_asset_dir / "ohlcv.npy", ohlcv)
    np.save(staging_asset_dir / "dates.npy", shared_dates.to_numpy(dtype="datetime64[D]"))
    np.save(staging_asset_dir / "regimes.npy", regime_series.to_numpy(dtype=int))
    np.save(staging_asset_dir / "micro_indicators_raw.npy", micro_frame.values.astype(float))
    np.save(staging_asset_dir / "micro_indicators.npy", micro_scaled)
    np.save(staging_asset_dir / "macro_indicators_raw.npy", macro_frame.values.astype(float))
    np.save(staging_asset_dir / "macro_indicators.npy", macro_scaled)
    (staging_asset_dir / "tickers.json").write_text(json.dumps(tickers, indent=2) + "\n")
    (staging_asset_dir / "feature_names.json").write_text(
        json.dumps(
            {
                "micro": list(micro_frame.columns),
                "macro": list(macro_frame.columns),
                "ohlcv_fields": list(OHLCV_FIELDS),
            },
            indent=2,
        )
        + "\n"
    )

    if raw_output_path is not None:
        save_raw_market_csv(normalized_frame.loc[shared_dates], raw_output_path)

    existing_action_dim = int(metadata.get("action_dim", len(tickers)))
    model_matches_universe = existing_action_dim in {len(tickers), len(tickers) + 1}
    use_signal_policy = bool(metadata) and existing_backend == "sb3" and not model_matches_universe

    metadata.update(
        {
            "asset_class": asset_class,
            "tickers": tickers,
            "ticker_count": len(tickers),
            "benchmark_ticker": benchmark,
            "feature_version": "ohlcv-v2",
            "uses_ohlcv_features": True,
            "micro_feature_count": int(micro_frame.shape[1]),
            "macro_feature_count": int(macro_frame.shape[1]),
            "feature_selection": {
                "micro_source": "micro_indicators_raw.npy",
                "micro_original_count": int(micro_frame.shape[1]),
                "micro_selected_count": int(micro_scaled.shape[1]),
                "micro_dropped_indices": [],
                "macro_source": "macro_indicators_raw.npy",
                "macro_original_count": int(macro_frame.shape[1]),
                "macro_selected_count": int(macro_scaled.shape[1]),
                "macro_dropped_indices": [],
            },
            "ohlcv_shape": list(ohlcv.shape),
            "aligned_rows": int(len(shared_dates)),
            "date_start": str(shared_dates[0].date()),
            "date_end": str(shared_dates[-1].date()),
            "macro_source_path": str(macro_path),
            "macro_last_available_date": str(macro_frame.index.max().date()),
            "market_data_source": "yfinance",
            "market_data_updated_at": datetime.now(UTC).isoformat(),
            "micro_features": [feature for ticker in tickers for feature in MICRO_FEATURES],
        }
    )
    if use_signal_policy:
        metadata.update(
            {
                "algorithm": "signal",
                "policy_backend": "single_agent_signal",
                "model_file": "signal_policy.json",
                "action_dim": len(tickers) + 1,
                "cash_enabled": True,
                "policy_replaced_reason": (
                    "Existing PPO action dimension did not match the rebuilt ticker universe. "
                    "Retrain PPO to replace this deterministic signal policy."
                ),
                "previous_policy_backend": existing_backend,
                "previous_action_dim": existing_action_dim,
            }
        )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    _commit_staged_artifact_dir(staging_asset_dir, asset_dir)

    return AssetDataset(
        asset_class=asset_class,
        tickers=tickers,
        dates=shared_dates,
        prices=prices,
        ohlcv=ohlcv,
        regimes=regime_series.to_numpy(dtype=int),
        micro_indicators_raw=micro_frame.values.astype(float),
        micro_indicators_scaled=micro_scaled,
        macro_indicators_raw=macro_frame.values.astype(float),
        macro_indicators_scaled=macro_scaled,
        metadata=metadata,
    )
