from __future__ import annotations

from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
import json
import shutil
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


def _backup_file(path: Path, *, timestamp: str) -> str | None:
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.stem}.backup_before_repair_{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return str(backup_path)


def _save_array_with_backup(path: Path, values: np.ndarray, *, timestamp: str) -> str | None:
    backup_path = _backup_file(path, timestamp=timestamp)
    np.save(path, values)
    return backup_path


def repair_asset(asset_class: str, *, artifact_root: Path, dry_run: bool) -> dict:
    asset_dir = artifact_root / asset_class
    tickers = json.loads((asset_dir / "tickers.json").read_text())
    prices = np.load(asset_dir / "prices.npy")
    dates_path = asset_dir / "dates.npy"
    dates = np.load(dates_path) if dates_path.exists() else None
    ohlcv_path = asset_dir / "ohlcv.npy"
    ohlcv = np.load(ohlcv_path) if ohlcv_path.exists() else None
    regimes = np.load(asset_dir / "regimes.npy")
    micro = np.load(asset_dir / "micro_indicators.npy")
    macro = np.load(asset_dir / "macro_indicators.npy")
    micro_raw_path = asset_dir / "micro_indicators_raw.npy"
    macro_raw_path = asset_dir / "macro_indicators_raw.npy"
    micro_raw = np.load(micro_raw_path) if micro_raw_path.exists() else None
    macro_raw = np.load(macro_raw_path) if macro_raw_path.exists() else None

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
    if micro_raw is not None:
        micro_raw = np.asarray(micro_raw, dtype=float)[-alignment.aligned_rows:]
    if macro_raw is not None:
        macro_raw = np.asarray(macro_raw, dtype=float)[-alignment.aligned_rows:]

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_shapes = {
        "dates.npy": list(dates.shape),
        "prices.npy": list(prices.shape),
        "ohlcv.npy": list(ohlcv.shape),
        "regimes.npy": list(regimes.shape),
        "micro_indicators.npy": list(micro.shape),
        "macro_indicators.npy": list(macro.shape),
    }
    if micro_raw is not None:
        output_shapes["micro_indicators_raw.npy"] = list(micro_raw.shape)
    if macro_raw is not None:
        output_shapes["macro_indicators_raw.npy"] = list(macro_raw.shape)
    backups: dict[str, str] = {}
    if not dry_run:
        arrays_to_write = {
            "dates.npy": dates,
            "prices.npy": prices,
            "ohlcv.npy": ohlcv,
            "regimes.npy": regimes,
            "micro_indicators.npy": micro,
            "macro_indicators.npy": macro,
        }
        if micro_raw is not None:
            arrays_to_write["micro_indicators_raw.npy"] = micro_raw
        if macro_raw is not None:
            arrays_to_write["macro_indicators_raw.npy"] = macro_raw
        for filename, values in arrays_to_write.items():
            backup_path = _save_array_with_backup(
                asset_dir / filename,
                values,
                timestamp=timestamp,
            )
            if backup_path is not None:
                backups[filename] = backup_path

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
            "repair_dry_run": dry_run,
        }
    )
    if not dry_run:
        _save_json(asset_dir / "metadata.json", metadata)
    return {
        **metadata,
        "would_write": output_shapes,
        "array_backups": backups,
    }


def repair_meta(*, artifact_root: Path, dry_run: bool) -> dict:
    meta_dir = artifact_root / "meta"
    metadata = _load_json(meta_dir / "metadata.json", DEFAULT_META_METADATA)
    metadata.update({"repaired_at": datetime.now(UTC).isoformat(), "repair_dry_run": dry_run})
    if not dry_run:
        _save_json(meta_dir / "metadata.json", metadata)
    return metadata


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Repair processed Foresight artifacts by aligning row counts.")
    parser.add_argument(
        "--artifact-root",
        default=str(ARTIFACT_ROOT),
        help="Processed artifact root to repair.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the alignment changes without writing arrays or metadata.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_root = Path(args.artifact_root)
    reports = {
        asset_class: repair_asset(
            asset_class,
            artifact_root=artifact_root,
            dry_run=args.dry_run,
        )
        for asset_class in ASSET_CLASSES
    }
    reports["meta"] = repair_meta(artifact_root=artifact_root, dry_run=args.dry_run)
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
