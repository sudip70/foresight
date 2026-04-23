from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import json
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ml.artifacts import ASSET_CLASSES, validate_and_align_asset_artifacts


ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "processed"


DEFAULT_ASSET_METADATA = {
    "stock": {"algorithm": "ppo", "policy_backend": "sb3", "model_file": "model.zip"},
    "crypto": {"algorithm": "ppo", "policy_backend": "sb3", "model_file": "model.zip"},
    "etf": {"algorithm": "ppo", "policy_backend": "sb3", "model_file": "model.zip"},
}
DEFAULT_META_METADATA = {
    "algorithm": "sac",
    "policy_backend": "sb3",
    "model_file": "model.zip",
    "feature_version": "legacy-rl-v1",
}


def _load_json(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return dict(fallback)
    data = json.loads(path.read_text())
    merged = dict(fallback)
    merged.update(data)
    return merged


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def repair_asset(asset_class: str) -> dict:
    asset_dir = ARTIFACT_ROOT / asset_class
    tickers = json.loads((asset_dir / "tickers.json").read_text())
    prices = np.load(asset_dir / "prices.npy")
    dates_path = asset_dir / "dates.npy"
    dates = np.load(dates_path) if dates_path.exists() else None
    ohlcv_path = asset_dir / "ohlcv.npy"
    ohlcv = np.load(ohlcv_path) if ohlcv_path.exists() else None
    regimes = np.load(asset_dir / "regimes.npy")
    micro = np.load(asset_dir / "micro_indicators.npy")
    macro = np.load(asset_dir / "macro_indicators.npy")

    dates, prices, ohlcv, regimes, micro, macro, alignment = validate_and_align_asset_artifacts(
        tickers=tickers,
        dates=dates,
        prices=prices,
        ohlcv=ohlcv,
        regimes=regimes,
        micro_indicators=micro,
        macro_indicators=macro,
        strict=False,
    )

    np.save(asset_dir / "dates.npy", dates)
    np.save(asset_dir / "prices.npy", prices)
    np.save(asset_dir / "ohlcv.npy", ohlcv)
    np.save(asset_dir / "regimes.npy", regimes)
    np.save(asset_dir / "micro_indicators.npy", micro)
    np.save(asset_dir / "macro_indicators.npy", macro)

    metadata = _load_json(asset_dir / "metadata.json", DEFAULT_ASSET_METADATA[asset_class])
    metadata.update(
        {
            "asset_class": asset_class,
            "aligned_rows": int(prices.shape[0]),
            "ticker_count": len(tickers),
            "micro_feature_count": int(micro.shape[1]),
            "macro_feature_count": int(macro.shape[1]),
            "feature_version": metadata.get("feature_version", "legacy-rl-v1"),
            "original_rows": alignment.original_rows,
            "trimmed": alignment.trimmed,
            "repaired_at": datetime.now(UTC).isoformat(),
        }
    )
    _save_json(asset_dir / "metadata.json", metadata)
    return metadata


def repair_meta() -> dict:
    meta_dir = ARTIFACT_ROOT / "meta"
    metadata = _load_json(meta_dir / "metadata.json", DEFAULT_META_METADATA)
    metadata.update({"repaired_at": datetime.now(UTC).isoformat()})
    _save_json(meta_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    reports = {asset_class: repair_asset(asset_class) for asset_class in ASSET_CLASSES}
    reports["meta"] = repair_meta()
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
