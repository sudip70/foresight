from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from backend.app.ml.envs import meta_observation_dim, single_agent_observation_dim


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _asset_prices(rows: int, offset: float) -> np.ndarray:
    index = np.arange(rows, dtype=float)
    first = 50 + offset + (index * 0.8) + (np.sin(index / 3) * 0.5)
    second = 70 + offset + (index * 0.6) + (np.cos(index / 4) * 0.75)
    return np.column_stack([first, second])


def _asset_micro(rows: int, offset: float) -> np.ndarray:
    index = np.arange(rows, dtype=float)
    return np.column_stack(
        [
            np.sin(index / 4) + offset,
            np.cos(index / 5) + offset / 2,
            (index / rows) + offset / 10,
            np.sqrt(index + 1) / 5 + offset / 20,
        ]
    )


def _asset_macro(rows: int, offset: float) -> np.ndarray:
    index = np.arange(rows, dtype=float)
    return np.column_stack(
        [
            0.2 + (index / (rows * 2)) + offset / 50,
            0.4 + np.sin(index / 6) / 10 + offset / 60,
        ]
    )


def build_fixture_artifact_tree(tmp_path: Path, *, rows: int = 24) -> Path:
    artifact_root = tmp_path / "processed"
    macro_names = ["vix_market_volatility", "federal_funds_rate"]

    asset_specs = {
        "stock": {
            "tickers": ["AAPL", "MSFT"],
            "weights": [0.55, 0.45],
            "offset": 0.0,
        },
        "crypto": {
            "tickers": ["BTC-USD", "ETH-USD"],
            "weights": [0.65, 0.35],
            "offset": 5.0,
        },
        "etf": {
            "tickers": ["SPY", "QQQ"],
            "weights": [0.52, 0.48],
            "offset": 10.0,
        },
    }

    total_assets = sum(len(spec["tickers"]) for spec in asset_specs.values())
    per_asset_obs_dim = single_agent_observation_dim(
        n_assets=len(next(iter(asset_specs.values()))["tickers"]),
        micro_dim=4,
        macro_dim=2,
    )
    combined_micro_dim = 4 * len(asset_specs)
    combined_macro_dim = 2 * len(asset_specs)
    meta_obs_dim = meta_observation_dim(
        n_assets=total_assets,
        micro_dim=combined_micro_dim,
        macro_dim=combined_macro_dim,
    )

    for asset_class, spec in asset_specs.items():
        asset_dir = artifact_root / asset_class
        asset_dir.mkdir(parents=True, exist_ok=True)
        prices = _asset_prices(rows, spec["offset"])
        dates = np.arange(
            np.datetime64("2025-01-01"),
            np.datetime64("2025-01-01") + rows,
            dtype="datetime64[D]",
        )
        np.save(asset_dir / "prices.npy", prices)
        np.save(asset_dir / "dates.npy", dates)
        np.save(asset_dir / "regimes.npy", (np.arange(rows) + int(spec["offset"])) % 3)
        np.save(asset_dir / "micro_indicators.npy", _asset_micro(rows, spec["offset"]))
        np.save(asset_dir / "macro_indicators.npy", _asset_macro(rows, spec["offset"]))
        (asset_dir / "tickers.json").write_text(json.dumps(spec["tickers"]))
        _write_json(
            asset_dir / "model.json",
            {"weights": spec["weights"], "observation_dim": per_asset_obs_dim},
        )
        _write_json(
            asset_dir / "metadata.json",
            {
                "asset_class": asset_class,
                "feature_version": "fixture-v1",
                "policy_backend": "fixed",
                "algorithm": "ppo",
                "model_file": "model.json",
                "macro_feature_names": macro_names,
            },
        )

    meta_dir = artifact_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    weights_matrix = np.zeros((total_assets, meta_obs_dim), dtype=float)
    for index in range(total_assets):
        weights_matrix[index, index] = 1.2
        weights_matrix[index, total_assets * 2 + index] = 0.35
        weights_matrix[index, total_assets * 3 + index] = 0.20
        weights_matrix[index, (total_assets * 4) + 11 + (index % combined_micro_dim)] = 0.08
    bias = np.linspace(0.1, 0.6, total_assets)
    _write_json(
        meta_dir / "model.json",
        {
            "weights_matrix": weights_matrix.tolist(),
            "bias": bias.tolist(),
            "observation_dim": meta_obs_dim,
        },
    )
    _write_json(
        meta_dir / "metadata.json",
        {
            "feature_version": "fixture-v1",
            "policy_backend": "linear",
            "algorithm": "sac",
            "model_file": "model.json",
        },
    )

    return artifact_root
