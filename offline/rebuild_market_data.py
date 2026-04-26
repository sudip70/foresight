from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline.market_pipeline import (
    ASSET_UNIVERSES,
    build_asset_dataset,
    default_end_date,
    download_ohlcv_history,
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Refresh Foresight market data using real OHLCV inputs.")
    parser.add_argument(
        "--asset-class",
        action="append",
        choices=sorted(ASSET_UNIVERSES.keys()),
        dest="asset_classes",
        help="Asset class to refresh. Defaults to all asset classes.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(REPO_ROOT / "artifacts" / "processed"),
        help="Directory where processed artifact bundles are written.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "datasets" / "raw"),
        help="Directory where raw market datasets are written.",
    )
    parser.add_argument(
        "--macro-path",
        default=str(REPO_ROOT / "datasets" / "raw" / "macroeconomic_data_2010_2024.csv"),
        help="CSV file containing macroeconomic features.",
    )
    parser.add_argument(
        "--end-date",
        default=default_end_date(),
        help="Exclusive end date used for market downloads.",
    )
    parser.add_argument(
        "--fit-new-scalers",
        action="store_true",
        help="Fit new scalers instead of reusing the existing artifact scalers.",
    )
    parser.add_argument(
        "--skip-scaler-write",
        action="store_true",
        help="Transform data but do not overwrite scaler pickle files.",
    )
    return parser


def refresh_asset_class(
    *,
    asset_class: str,
    artifact_root: Path,
    dataset_root: Path,
    macro_path: Path,
    end_date: str,
    fit_new_scalers: bool,
    skip_scaler_write: bool,
    market_frame=None,
) -> dict:
    config = ASSET_UNIVERSES[asset_class]
    tickers = list(config["tickers"])
    benchmark = config["benchmark"]
    download_tickers = tickers if benchmark in tickers else [*tickers, benchmark]

    if market_frame is None:
        market_frame = download_ohlcv_history(
            download_tickers,
            start_date=config["start_date"],
            end_date=end_date,
        )
    dataset = build_asset_dataset(
        asset_class=asset_class,
        market_frame=market_frame,
        macro_path=macro_path,
        artifact_root=artifact_root,
        raw_output_path=dataset_root / "market" / f"{asset_class}_ohlcv.csv",
        reuse_existing_scalers=not fit_new_scalers,
        persist_scalers=not skip_scaler_write,
    )
    return {
        "rows": dataset.metadata["aligned_rows"],
        "date_start": dataset.metadata["date_start"],
        "date_end": dataset.metadata["date_end"],
        "benchmark_ticker": dataset.metadata["benchmark_ticker"],
        "uses_ohlcv_features": dataset.metadata["uses_ohlcv_features"],
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    asset_classes = args.asset_classes or sorted(ASSET_UNIVERSES.keys())
    artifact_root = Path(args.artifact_root)
    dataset_root = Path(args.dataset_root)
    macro_path = Path(args.macro_path)

    reports = {}
    for asset_class in asset_classes:
        reports[asset_class] = refresh_asset_class(
            asset_class=asset_class,
            artifact_root=artifact_root,
            dataset_root=dataset_root,
            macro_path=macro_path,
            end_date=args.end_date,
            fit_new_scalers=args.fit_new_scalers,
            skip_scaler_write=args.skip_scaler_write,
        )

    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
