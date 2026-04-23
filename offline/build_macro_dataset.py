from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import os

from fredapi import Fred
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "raw" / "macroeconomic_data_2010_2024.csv"


SERIES_MAP = {
    "FEDFUNDS": "Federal Funds Rate",
    "DGS10": "10-Year Treasury Yield",
    "CPIAUCSL": "CPI All Items",
    "PCE": "Personal Consumption Expenditures",
    "UNRATE": "Unemployment Rate",
    "INDPRO": "Industrial Production Index",
    "GDP": "GDP",
    "USRECD": "Recession Indicator",
    "VIXCLS": "VIX Market Volatility",
}


def build_macro_dataset(start_date: str, end_date: str, output_path: Path) -> Path:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is required to build macro data")

    fred = Fred(api_key=api_key)
    frame = pd.DataFrame()
    for series_id, display_name in SERIES_MAP.items():
        frame[display_name] = fred.get_series(series_id, start_date, end_date)

    frame = frame.ffill()
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "Date"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=True, index_label="Date")
    return output_path


def main() -> None:
    end_date = (date.today() + timedelta(days=1)).isoformat()
    output = build_macro_dataset(
        start_date="2010-01-01",
        end_date=end_date,
        output_path=DEFAULT_OUTPUT,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
