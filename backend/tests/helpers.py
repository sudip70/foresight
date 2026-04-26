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


def build_fixture_artifact_tree(
    tmp_path: Path,
    *,
    version: str = "v1",
    rows: int | None = None,
) -> Path:
    if version == "v3":
        return _build_v3_fixture_artifact_tree(tmp_path, rows=36 if rows is None else rows)
    if version != "v1":
        raise ValueError(f"Unsupported fixture artifact version: {version}")
    rows = 24 if rows is None else rows
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

    risky_assets = sum(len(spec["tickers"]) for spec in asset_specs.values())
    total_assets = risky_assets + 1
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
    for index in range(risky_assets):
        weights_matrix[index, index] = 1.2
        weights_matrix[index, total_assets * 2 + index] = 0.35
        weights_matrix[index, total_assets * 3 + index] = 0.20
        weights_matrix[index, (total_assets * 4) + 11 + (index % combined_micro_dim)] = 0.08
    weights_matrix[-1, total_assets * 3 + risky_assets] = 0.8
    bias = np.linspace(0.1, 0.7, total_assets)
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


def _build_v3_fixture_artifact_tree(tmp_path: Path, *, rows: int = 36) -> Path:
    artifact_root = tmp_path / "processed_v3"
    macro_names = ["vix_market_volatility", "federal_funds_rate"]
    shared_macro = _asset_macro(rows, 0.0)

    asset_specs = {
        "stock": {
            "tickers": ["AAPL", "MSFT"],
            "weights": [0.68, 0.20, 0.12],
            "offset": 0.0,
        },
        "crypto": {
            "tickers": ["BTC-USD", "ETH-USD"],
            "weights": [0.22, 0.43, 0.35],
            "offset": 4.0,
        },
        "etf": {
            "tickers": ["SPY", "QQQ"],
            "weights": [0.40, 0.42, 0.18],
            "offset": 8.0,
        },
    }

    risky_assets = sum(len(spec["tickers"]) for spec in asset_specs.values())
    total_assets = risky_assets + 1
    per_asset_obs_dim = single_agent_observation_dim(
        n_assets=3,
        micro_dim=4,
        macro_dim=2,
    )
    combined_micro_dim = 4 * len(asset_specs)
    meta_obs_dim = meta_observation_dim(
        n_assets=total_assets,
        micro_dim=combined_micro_dim,
        macro_dim=2,
        class_feature_dim=12,
    )
    dates = np.arange(
        np.datetime64("2025-01-01"),
        np.datetime64("2025-01-01") + rows,
        dtype="datetime64[D]",
    )

    for asset_class, spec in asset_specs.items():
        asset_dir = artifact_root / asset_class
        asset_dir.mkdir(parents=True, exist_ok=True)
        np.save(asset_dir / "prices.npy", _asset_prices(rows, spec["offset"]))
        np.save(asset_dir / "dates.npy", dates)
        np.save(asset_dir / "regimes.npy", (np.arange(rows) + int(spec["offset"])) % 3)
        np.save(asset_dir / "micro_indicators.npy", _asset_micro(rows, spec["offset"]))
        np.save(asset_dir / "macro_indicators.npy", shared_macro)
        np.save(asset_dir / "macro_indicators_raw.npy", shared_macro)
        (asset_dir / "tickers.json").write_text(json.dumps(spec["tickers"]))
        _write_json(
            asset_dir / "model.json",
            {"weights": spec["weights"], "observation_dim": per_asset_obs_dim},
        )
        _write_json(
            asset_dir / "metadata.json",
            {
                "asset_class": asset_class,
                "feature_version": "ohlcv-stationary-v4-cash-sleeve",
                "policy_backend": "fixed",
                "algorithm": "ppo",
                "model_file": "model.json",
                "macro_feature_names": macro_names,
                "action_dim": 3,
                "ppo_training_config": {
                    "cash_enabled": True,
                    "max_asset_weight": 0.85,
                    "max_cash_weight": 0.95,
                    "cash_annual_return": 0.04,
                },
            },
        )

    meta_dir = artifact_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    weights_matrix = np.zeros((total_assets, meta_obs_dim), dtype=float)
    sub_agent_offset = total_assets * 3
    class_context_offset = total_assets * 4
    portfolio_state_offset = class_context_offset + 12
    micro_offset = portfolio_state_offset + 2
    macro_offset = micro_offset + combined_micro_dim
    cash_index = total_assets - 1
    for index in range(risky_assets):
        weights_matrix[index, index] = 0.8
        weights_matrix[index, sub_agent_offset + index] = 1.1
        weights_matrix[index, total_assets * 2 + index] = 0.15
        weights_matrix[index, micro_offset + (index % combined_micro_dim)] = 0.05
    weights_matrix[cash_index, cash_index] = 0.4
    weights_matrix[cash_index, sub_agent_offset + cash_index] = 1.4
    weights_matrix[cash_index, class_context_offset + 3] = 0.3
    weights_matrix[cash_index, class_context_offset + 7] = 0.3
    weights_matrix[cash_index, class_context_offset + 11] = 0.3
    weights_matrix[cash_index, macro_offset] = 0.05
    bias = np.linspace(0.1, 0.5, total_assets)
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
            "feature_version": "sac-meta-v3-globalmacro-cashaware",
            "policy_backend": "linear",
            "algorithm": "sac",
            "model_file": "model.json",
            "class_feature_dim": 12,
            "uses_shared_macro": True,
            "meta_macro_feature_names": macro_names,
        },
    )

    return artifact_root
