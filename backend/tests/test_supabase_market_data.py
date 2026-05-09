from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

from backend.app.core.config import reset_settings
import backend.app.main as app_main
from backend.app.market.forecasting import SupabaseForecastEngine
from backend.app.market.repository import (
    InMemoryMarketDataRepository,
    MarketDataUnavailable,
    SupabaseMarketDataRepository,
)
from backend.app.ml.pipeline import reset_engine
from backend.tests.helpers import build_fixture_artifact_tree
from offline.supabase_refresh import run_refresh


def _assets():
    return [
        {
            "ticker": "AAA",
            "asset_class": "stock",
            "display_name": "AAA Corp",
            "provider_symbol": "AAA",
            "exchange": "NASDAQ",
            "currency": "USD",
            "sector": "Technology",
            "industry": "Software",
            "country": "US",
            "benchmark_group": "test",
            "min_history_days": 30,
            "active": True,
            "start_date": "2025-01-01",
        },
        {
            "ticker": "BBB",
            "asset_class": "etf",
            "display_name": "BBB ETF",
            "provider_symbol": "BBB",
            "exchange": "NYSEARCA",
            "currency": "USD",
            "sector": "ETF",
            "industry": "Broad Market",
            "country": "US",
            "benchmark_group": "test",
            "min_history_days": 30,
            "active": True,
            "start_date": "2025-01-01",
        },
    ]


def _ohlcv_rows(ticker="AAA", days=80):
    start = date(2025, 1, 1)
    rows = []
    for index in range(days):
        close = 100.0 + index * 0.3
        rows.append(
            {
                "ticker": ticker,
                "date": (start + timedelta(days=index)).isoformat(),
                "open": close - 0.2,
                "high": close + 0.6,
                "low": close - 0.7,
                "close": close,
                "adjusted_close": close,
                "volume": 1_000_000 + index,
                "provider": "fake",
            }
        )
    return rows


def _proxy_ohlcv_rows(ticker: str, days: int = 80):
    rows = _ohlcv_rows(ticker=ticker, days=days)
    for index, row in enumerate(rows):
        close = 400.0 + index
        row.update(
            {
                "open": close - 1.0,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "adjusted_close": close,
                "volume": 10_000_000 + index,
            }
        )
    return rows


def _seed_repository() -> InMemoryMarketDataRepository:
    repository = InMemoryMarketDataRepository()
    repository.upsert_universe(
        [
            {
                key: value
                for key, value in asset.items()
                if key
                in {
                    "ticker",
                    "asset_class",
                    "display_name",
                    "provider_symbol",
                    "exchange",
                    "currency",
                    "sector",
                    "industry",
                    "country",
                    "benchmark_group",
                    "min_history_days",
                    "active",
                }
            }
            for asset in _assets()
        ]
    )
    repository.upsert_ohlcv(_ohlcv_rows("AAA"))
    repository.upsert_profiles(
        [
            {
                "ticker": "AAA",
                "as_of_date": "2025-03-21",
                "market_cap": None,
                "pe_ratio": None,
                "last_sale": 123.0,
                "exchange": "NASDAQ",
            }
        ]
    )
    return repository


class FakeProvider:
    name = "fake"

    def fetch_daily_ohlcv(
        self,
        tickers,
        *,
        start_date,
        end_date,
        batch_size=10,
        freshness_days=10,
    ):
        return _ohlcv_rows("AAA")

    def fetch_company_profiles(self, tickers):
        return [
            {
                "ticker": row["ticker"],
                "as_of_date": "2025-03-21",
                "market_cap": None,
                "pe_ratio": None,
                "last_sale": 123.0,
                "exchange": row.get("exchange"),
            }
            for row in tickers
        ]

    def fetch_macro_observations(self, *, start_date, end_date):
        return [
            {
                "date": "2025-03-21",
                "vix": 18.0,
                "federal_funds_rate": 4.0,
                "treasury_10y": 4.2,
                "unemployment_rate": 4.1,
                "cpi_all_items": 320.0,
                "recession_indicator": 0.0,
                "provider": "fake",
            }
        ]

    def fetch_market_indices(self, indices, *, start_date, end_date):
        return [
            {
                "symbol": row["symbol"],
                "as_of_date": "2025-03-21",
                "label": row.get("label", row["symbol"]),
                "display_name": row.get("display_name", row["symbol"]),
                "provider_symbol": row.get("provider_symbol", row["symbol"]),
                "value": 5000.0,
                "previous_close": 4975.0,
                "change": 25.0,
                "change_percent": 25.0 / 4975.0,
                "currency": row.get("currency", "USD"),
                "provider": "fake",
                "display_order": int(row.get("display_order") or 0),
            }
            for row in indices
        ]


def test_in_memory_repository_upsert_is_idempotent():
    repository = InMemoryMarketDataRepository()
    repository.upsert_universe([_assets()[0]])
    repository.upsert_ohlcv([_ohlcv_rows("AAA", days=1)[0]])
    repository.upsert_ohlcv([_ohlcv_rows("AAA", days=1)[0]])

    assert len(repository.tables["market_ohlcv_daily"]) == 1
    assert repository.coverage_for_ticker("AAA")["row_count"] == 1


def test_supabase_repository_paginates_ohlcv_history():
    class PagedRepository(SupabaseMarketDataRepository):
        def __init__(self):
            super().__init__(url="https://example.supabase.co", service_role_key="test")
            self.rows = _ohlcv_rows("AAA", days=1205)

        def _get(self, table, params=None):
            if table != "market_ohlcv_daily":
                return []
            offset = int((params or {}).get("offset", 0))
            limit = int((params or {}).get("limit", 1000))
            return self.rows[offset : offset + limit]

    repository = PagedRepository()

    history = repository.get_ohlcv_history("AAA")
    coverage = repository.coverage_for_ticker("AAA")

    assert len(history) == 1205
    assert coverage["row_count"] == 1205
    assert coverage["latest_date"] == history[-1]["date"]


def test_refresh_deactivates_tickers_removed_from_universe_config():
    repository = InMemoryMarketDataRepository()
    old_asset = {
        **_assets()[0],
        "ticker": "OLD",
        "display_name": "Removed Asset",
        "provider_symbol": "OLD",
    }
    repository.upsert_universe(_assets() + [old_asset])

    run_refresh(
        repository=repository,
        provider=FakeProvider(),
        assets=_assets(),
        start_date="2025-01-01",
        end_date="2025-04-01",
        horizons=[30],
        window_size=20,
        freshness_days=500,
        dry_run=False,
    )

    active_tickers = {row["ticker"] for row in repository.list_universe()}
    removed_row = next(row for row in repository.tables["asset_universe"] if row["ticker"] == "OLD")
    assert "OLD" not in active_tickers
    assert removed_row["active"] is False


def test_refresh_job_upserts_market_index_snapshots():
    repository = InMemoryMarketDataRepository()
    report = run_refresh(
        repository=repository,
        provider=FakeProvider(),
        assets=_assets(),
        market_indices=[
            {
                "symbol": "SP500",
                "label": "S&P 500",
                "display_name": "S&P 500 Index",
                "provider_symbol": "^GSPC",
                "currency": "USD",
                "display_order": 1,
            }
        ],
        start_date="2025-01-01",
        end_date="2025-04-01",
        horizons=[30],
        window_size=20,
        freshness_days=500,
        dry_run=False,
    )

    assert report["rows"]["market_index_snapshots"] == 1
    assert repository.list_latest_market_indices()[0]["symbol"] == "SP500"


def test_inactive_tickers_are_ignored_by_market_forecast_snapshots():
    repository = _seed_repository()
    repository.upsert_universe(
        [
            {
                **_assets()[0],
                "ticker": "OLD",
                "display_name": "Removed Asset",
                "provider_symbol": "OLD",
                "active": False,
            }
        ]
    )
    repository.upsert_ohlcv(_ohlcv_rows("OLD"))
    engine = SupabaseForecastEngine(repository, resettable_settings())
    active_forecast = engine.build_ticker_forecast(
        ticker="AAA",
        horizon_days=30,
        window_size=20,
        prefer_snapshot=False,
    )
    inactive_snapshot = {
        **active_forecast,
        "ticker": "OLD",
        "returns": {"bear": 0.5, "base": 2.0, "bull": 3.0},
        "target_prices": {"bear": 150.0, "base": 300.0, "bull": 400.0},
    }
    repository.upsert_forecasts(
        [
            engine.forecast_snapshot_row(active_forecast),
            engine.forecast_snapshot_row(inactive_snapshot),
        ]
    )

    market = engine.run_market_forecast(horizon_days=30, window_size=20, risk=1.0, top_n=10)

    assert {row["ticker"] for row in market["ranked_tickers"]} == {"AAA"}


def test_market_forecast_ignores_stale_snapshots_and_recomputes_from_latest_history():
    repository = _seed_repository()
    engine = SupabaseForecastEngine(repository, resettable_settings())
    forecast = engine.build_ticker_forecast(
        ticker="AAA",
        horizon_days=30,
        window_size=20,
        prefer_snapshot=False,
    )
    stale_snapshot = {
        **engine.forecast_snapshot_row(forecast),
        "as_of_date": "2025-01-30",
        "latest_price": 1.0,
        "base_return": 9.0,
    }
    repository.upsert_forecasts([stale_snapshot])

    market = engine.run_market_forecast(horizon_days=30, window_size=20, risk=0.5, top_n=10)

    assert market["ranked_tickers"][0]["source"] == "supabase_ohlcv"
    assert market["ranked_tickers"][0]["latest_date"] == repository.coverage_for_ticker("AAA")[
        "latest_date"
    ]


def test_forecast_engine_refreshes_cached_ticker_rows_after_ohlcv_update():
    repository = _seed_repository()
    engine = SupabaseForecastEngine(repository, resettable_settings())

    original = engine.run_ticker_forecast(ticker="AAA", horizon_days=30, window_size=20)
    new_row = _ohlcv_rows("AAA", days=81)[-1]
    new_row.update(
        {
            "close": 250.0,
            "adjusted_close": 250.0,
            "open": 249.0,
            "high": 252.0,
            "low": 248.0,
        }
    )
    repository.upsert_ohlcv([new_row])

    refreshed = engine.run_ticker_forecast(ticker="AAA", horizon_days=30, window_size=20)

    assert original["latest_date"] == "2025-03-21"
    assert refreshed["latest_date"] == new_row["date"]
    assert refreshed["latest_price"] == 250.0


def test_forecast_engine_refreshes_same_date_rows_after_refresh_run_changes():
    repository = _seed_repository()
    engine = SupabaseForecastEngine(repository, resettable_settings())

    original = engine.run_ticker_forecast(ticker="AAA", horizon_days=30, window_size=20)
    revised_latest_row = _ohlcv_rows("AAA", days=80)[-1]
    revised_latest_row.update(
        {
            "close": 275.0,
            "adjusted_close": 275.0,
            "open": 274.0,
            "high": 277.0,
            "low": 273.0,
        }
    )
    repository.upsert_ohlcv([revised_latest_row])
    repository.upsert_refresh_run(
        {
            "id": "same-date-refresh",
            "provider": "fake",
            "status": "completed",
            "started_at": "2025-03-22T13:35:00Z",
            "finished_at": "2025-03-22T13:36:00Z",
        }
    )

    refreshed = engine.run_ticker_forecast(ticker="AAA", horizon_days=30, window_size=20)

    assert original["latest_date"] == revised_latest_row["date"]
    assert original["latest_price"] != 275.0
    assert refreshed["latest_date"] == revised_latest_row["date"]
    assert refreshed["latest_price"] == 275.0


def test_market_forecast_cache_refreshes_after_ohlcv_update():
    repository = _seed_repository()
    engine = SupabaseForecastEngine(repository, resettable_settings())

    original = engine.run_market_forecast(horizon_days=30, window_size=20, risk=0.5, top_n=10)
    new_row = _ohlcv_rows("AAA", days=81)[-1]
    new_row.update(
        {
            "close": 300.0,
            "adjusted_close": 300.0,
            "open": 299.0,
            "high": 302.0,
            "low": 298.0,
        }
    )
    repository.upsert_ohlcv([new_row])

    refreshed = engine.run_market_forecast(horizon_days=30, window_size=20, risk=0.5, top_n=10)
    original_by_ticker = {row["ticker"]: row for row in original["ranked_tickers"]}
    refreshed_by_ticker = {row["ticker"]: row for row in refreshed["ranked_tickers"]}

    assert original_by_ticker["AAA"]["latest_date"] == "2025-03-21"
    assert refreshed_by_ticker["AAA"]["latest_date"] == new_row["date"]
    assert refreshed_by_ticker["AAA"]["latest_price"] == 300.0


def test_universe_and_macro_payloads_refresh_after_repository_updates():
    repository = _seed_repository()
    repository.upsert_macro(
        [
            {
                "date": "2025-03-21",
                "vix": 18.0,
                "federal_funds_rate": 4.0,
                "treasury_10y": 4.2,
                "unemployment_rate": 4.1,
                "cpi_all_items": 320.0,
                "recession_indicator": 0.0,
                "provider": "fake",
            }
        ]
    )
    engine = SupabaseForecastEngine(repository, resettable_settings())

    original_universe = engine.universe_payload()
    original_market = engine.run_market_forecast(
        horizon_days=30,
        window_size=20,
        risk=0.5,
        top_n=10,
    )
    repository.upsert_universe(
        [
            {
                **_assets()[0],
                "ticker": "CCC",
                "display_name": "CCC Corp",
                "provider_symbol": "CCC",
                "active": True,
            }
        ]
    )
    repository.upsert_macro(
        [
            {
                "date": "2025-03-22",
                "vix": 19.0,
                "federal_funds_rate": 4.0,
                "treasury_10y": 4.2,
                "unemployment_rate": 4.1,
                "cpi_all_items": 320.0,
                "recession_indicator": 0.0,
                "provider": "fake",
            }
        ]
    )

    refreshed_universe = engine.universe_payload()
    refreshed_market = engine.run_market_forecast(
        horizon_days=30,
        window_size=20,
        risk=0.5,
        top_n=10,
    )

    assert "CCC" not in {row["ticker"] for row in original_universe["tickers"]}
    assert "CCC" in {row["ticker"] for row in refreshed_universe["tickers"]}
    assert original_market["macro_snapshot"]["date"] == "2025-03-21"
    assert refreshed_market["macro_snapshot"]["date"] == "2025-03-22"


def test_market_forecast_fills_tickers_missing_from_partial_snapshot_cache():
    repository = InMemoryMarketDataRepository()
    repository.upsert_universe(
        [
            *_assets(),
            {
                "ticker": "CCC",
                "asset_class": "crypto",
                "display_name": "CCC Crypto",
                "provider_symbol": "CCC-USD",
                "exchange": "CRYPTO",
                "currency": "USD",
                "sector": "Crypto",
                "industry": "Digital Asset",
                "country": "Global",
                "benchmark_group": "test",
                "min_history_days": 30,
                "active": True,
            },
        ]
    )
    repository.upsert_ohlcv(_ohlcv_rows("AAA"))
    repository.upsert_ohlcv(_ohlcv_rows("BBB"))
    repository.upsert_ohlcv(_ohlcv_rows("CCC"))
    engine = SupabaseForecastEngine(repository, resettable_settings())
    crypto_forecast = engine.build_ticker_forecast(
        ticker="CCC",
        horizon_days=30,
        window_size=20,
        prefer_snapshot=False,
    )
    repository.upsert_forecasts([engine.forecast_snapshot_row(crypto_forecast)])

    market = engine.run_market_forecast(horizon_days=30, window_size=20, risk=0.5, top_n=10)

    by_ticker = {row["ticker"]: row for row in market["ranked_tickers"]}
    assert set(by_ticker) == {"AAA", "BBB", "CCC"}
    assert by_ticker["CCC"]["source"] == "supabase_forecast_snapshot"
    assert by_ticker["AAA"]["source"] == "supabase_ohlcv"
    assert by_ticker["BBB"]["source"] == "supabase_ohlcv"
    expected_annualized = (
        (1.0 + by_ticker["CCC"]["returns"]["base"]) ** (252.0 / 30.0)
    ) - 1.0
    assert abs(by_ticker["CCC"]["risk_metrics"]["annualized_return"] - expected_annualized) < 1e-12


def test_refresh_job_logs_partial_failure_and_precomputes_forecasts():
    repository = InMemoryMarketDataRepository()
    report = run_refresh(
        repository=repository,
        provider=FakeProvider(),
        assets=_assets(),
        start_date="2025-01-01",
        end_date="2025-04-01",
        horizons=[30],
        window_size=20,
        freshness_days=500,
        dry_run=False,
    )

    assert report["status"] == "partial"
    assert len(repository.tables["forecast_snapshots"]) == 1
    failed_items = [
        item for item in repository.tables["refresh_run_items"] if item["status"] == "failed"
    ]
    assert any(item["ticker"] == "BBB" and item["stage"] == "ohlcv" for item in failed_items)


def test_forecast_engine_uses_stored_snapshot_for_default_horizon():
    repository = _seed_repository()
    engine = SupabaseForecastEngine(repository, resettable_settings())
    forecast = engine.build_ticker_forecast(
        ticker="AAA",
        horizon_days=30,
        window_size=20,
        prefer_snapshot=False,
    )
    snapshot_row = engine.forecast_snapshot_row(forecast)
    for path in snapshot_row["forecast_paths_json"].values():
        for point in path:
            point["date"] = (date(2025, 1, 1) + timedelta(days=point["day"])).isoformat()
    repository.upsert_forecasts([snapshot_row])

    stored = engine.run_ticker_forecast(ticker="AAA", horizon_days=30, window_size=20)

    assert stored["snapshot_used"] is True
    assert stored["source"] == "supabase_forecast_snapshot"
    today = date.today().isoformat()
    assert stored["forecast_start_date"] == today
    assert stored["forecast_paths"]["base"][0]["date"] == today
    assert stored["latest_date"] == repository.coverage_for_ticker("AAA")["latest_date"]
    expected_annualized = ((1.0 + stored["returns"]["base"]) ** (252.0 / 30.0)) - 1.0
    assert abs(stored["risk_metrics"]["annualized_return"] - expected_annualized) < 1e-12


def test_ticker_forecast_reports_change_since_previous_snapshot():
    repository = _seed_repository()
    engine = SupabaseForecastEngine(repository, resettable_settings())
    current_forecast = engine.build_ticker_forecast(
        ticker="AAA",
        horizon_days=30,
        window_size=20,
        prefer_snapshot=False,
    )
    previous_snapshot = {
        **engine.forecast_snapshot_row(current_forecast),
        "as_of_date": "2025-03-20",
        "base_return": current_forecast["returns"]["base"] - 0.05,
        "bear_return": current_forecast["returns"]["bear"] - 0.03,
        "bull_return": current_forecast["returns"]["bull"] - 0.07,
        "confidence": current_forecast["confidence"] - 0.10,
        "base_target": current_forecast["target_prices"]["base"] - 5.0,
    }
    repository.upsert_forecasts([previous_snapshot])

    refreshed = engine.run_ticker_forecast(ticker="AAA", horizon_days=30, window_size=20)

    assert refreshed["forecast_change"]["available"] is True
    assert refreshed["forecast_change"]["previous_as_of"] == "2025-03-20"
    assert round(refreshed["forecast_change"]["base_return_delta"], 6) == 0.05
    assert refreshed["data_quality"]["source"] == "supabase_ohlcv"
    assert refreshed["data_quality"]["history_rows"] == 80


def test_forecast_engine_ignores_snapshot_that_does_not_match_latest_market_date():
    repository = _seed_repository()
    engine = SupabaseForecastEngine(repository, resettable_settings())
    forecast = engine.build_ticker_forecast(
        ticker="AAA",
        horizon_days=30,
        window_size=20,
        prefer_snapshot=False,
    )
    stale_row = {
        **engine.forecast_snapshot_row(forecast),
        "as_of_date": "2025-01-30",
        "latest_price": 1.0,
    }
    repository.upsert_forecasts([stale_row])

    computed = engine.run_ticker_forecast(ticker="AAA", horizon_days=30, window_size=20)

    assert computed["snapshot_used"] is False
    assert computed["source"] == "supabase_ohlcv"


def test_portfolio_simulation_applies_constraints_and_explanations():
    repository = InMemoryMarketDataRepository()
    repository.upsert_universe(
        [
            *_assets(),
            {
                "ticker": "CCC",
                "asset_class": "crypto",
                "display_name": "CCC Crypto",
                "provider_symbol": "CCC-USD",
                "exchange": "CRYPTO",
                "currency": "USD",
                "sector": "Crypto",
                "industry": "Digital Asset",
                "country": "Global",
                "benchmark_group": "test",
                "min_history_days": 30,
                "active": True,
            },
        ]
    )
    repository.upsert_ohlcv(_ohlcv_rows("AAA"))
    repository.upsert_ohlcv(_ohlcv_rows("BBB"))
    repository.upsert_ohlcv(_ohlcv_rows("CCC"))
    engine = SupabaseForecastEngine(repository, resettable_settings())

    result = engine.run_portfolio_simulation(
        amount=10_000,
        risk=0.9,
        horizon_days=30,
        window_size=20,
        selected_tickers=["AAA", "BBB", "CCC"],
        max_crypto_weight=0.05,
        min_cash_weight=0.50,
        max_single_position_weight=0.40,
    )

    allocations = {allocation["ticker"]: allocation for allocation in result["asset_allocations"]}
    class_allocations = {
        allocation["asset_class"]: allocation for allocation in result["class_allocations"]
    }
    assert class_allocations["cash"]["weight"] >= 0.50
    assert class_allocations.get("crypto", {"weight": 0.0})["weight"] <= 0.0500001
    assert "min_cash_weight" in result["constraints_applied"]["binding"]
    assert "max_crypto_weight" in result["constraints_applied"]["binding"]
    assert result["allocation_explanations"]
    assert result["benchmark_comparison"][0]["ticker"] == "CASH"


def test_app_forecast_endpoints_work_from_market_repo_when_artifacts_are_broken(
    tmp_path,
    monkeypatch,
):
    repository = _seed_repository()
    artifact_root = build_fixture_artifact_tree(tmp_path)
    (artifact_root / "stock" / "prices.npy").unlink()
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    monkeypatch.setenv("FORESIGHT_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FORESIGHT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setattr(app_main, "build_market_repository", lambda settings: repository)
    reset_settings()
    reset_engine()

    with TestClient(app_main.create_app()) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        universe = client.get("/api/universe")
        assert universe.status_code == 200
        assert universe.json()["source"] == "supabase"

        forecast = client.post(
            "/api/forecasts/ticker",
            json={"ticker": "AAA", "horizon_days": 30, "window_size": 20},
        )
        assert forecast.status_code == 200
        assert forecast.json()["source"] == "supabase_ohlcv"

        profile = client.get("/api/tickers/AAA/profile")
        assert profile.status_code == 200
        assert profile.json()["fields"]["pe_ratio"] is None
        assert profile.json()["fields"]["sector"] == "Technology"
        assert profile.json()["fields"]["industry"] == "Software"
        assert profile.json()["fields"]["country"] == "US"

        repository.upsert_market_indices(
            [
                {
                    "symbol": "SP500",
                    "as_of_date": "2025-03-21",
                    "label": "S&P 500",
                    "display_name": "S&P 500 Index",
                    "provider_symbol": "^GSPC",
                    "value": 5000.0,
                    "previous_close": 4975.0,
                    "change": 25.0,
                    "change_percent": 25.0 / 4975.0,
                    "currency": "USD",
                    "provider": "fake",
                    "display_order": 1,
                }
            ]
        )
        indices = client.get("/api/market/indices")
        assert indices.status_code == 200
        assert indices.json()["indices"][0]["symbol"] == "SP500"

    reset_settings()
    reset_engine()


def test_market_indices_fall_back_to_supabase_proxy_history(tmp_path, monkeypatch):
    repository = InMemoryMarketDataRepository()
    repository.upsert_ohlcv(_proxy_ohlcv_rows("SPY"))
    repository.upsert_ohlcv(_proxy_ohlcv_rows("QQQ"))
    repository.upsert_ohlcv(_proxy_ohlcv_rows("DIA"))
    artifact_root = tmp_path / "missing-artifacts"
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    monkeypatch.setenv("FORESIGHT_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FORESIGHT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("FORESIGHT_REQUIRE_SUPABASE", "true")
    monkeypatch.setenv("FORESIGHT_LOAD_ARTIFACT_ENGINE", "false")
    monkeypatch.setattr(app_main, "build_market_repository", lambda settings: repository)
    monkeypatch.setattr(
        app_main,
        "fetch_market_index_snapshots",
        lambda settings, repository=None: (_ for _ in ()).throw(ValueError("provider failed")),
    )
    monkeypatch.setattr(
        app_main,
        "fetch_market_index_history",
        lambda settings, *, symbol, history_range: (_ for _ in ()).throw(
            ValueError("No index history returned")
        ),
    )
    reset_settings()
    reset_engine()

    with TestClient(app_main.create_app()) as client:
        indices = client.get("/api/market/indices")
        assert indices.status_code == 200
        assert indices.json()["source"] == "supabase_proxy"
        assert {row["symbol"] for row in indices.json()["indices"]} == {
            "SP500",
            "NASDAQ",
            "DOW",
        }

        history = client.get("/api/market/indices/SP500/history?range=1y")
        assert history.status_code == 200
        payload = history.json()
        assert payload["source"] == "supabase_proxy"
        assert payload["provider_symbol"] == "SPY"
        assert payload["summary"]["points"] > 1

    reset_settings()
    reset_engine()


def test_market_indices_provider_failure_without_repository_returns_503(tmp_path, monkeypatch):
    artifact_root = tmp_path / "missing-artifacts"
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    monkeypatch.setenv("FORESIGHT_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FORESIGHT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("FORESIGHT_LOAD_ARTIFACT_ENGINE", "false")
    monkeypatch.setattr(app_main, "build_market_repository", lambda settings: None)
    monkeypatch.setattr(
        app_main,
        "fetch_market_index_snapshots",
        lambda settings, repository=None: (_ for _ in ()).throw(ValueError("provider failed")),
    )
    reset_settings()
    reset_engine()

    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        response = client.get("/api/market/indices")

    assert response.status_code == 503
    assert "provider failed" in response.json()["detail"]
    assert "Supabase proxy fallback unavailable" in response.json()["detail"]

    reset_settings()
    reset_engine()


def test_market_index_history_provider_exception_returns_503(tmp_path, monkeypatch):
    artifact_root = tmp_path / "missing-artifacts"
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    monkeypatch.setenv("FORESIGHT_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FORESIGHT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("FORESIGHT_LOAD_ARTIFACT_ENGINE", "false")
    monkeypatch.setattr(app_main, "build_market_repository", lambda settings: None)
    monkeypatch.setattr(
        app_main,
        "fetch_market_index_history",
        lambda settings, *, symbol, history_range: (_ for _ in ()).throw(
            RuntimeError("provider exploded")
        ),
    )
    reset_settings()
    reset_engine()

    with TestClient(app_main.create_app(), raise_server_exceptions=False) as client:
        response = client.get("/api/market/indices/SP500/history?range=1y")

    assert response.status_code == 503
    assert "provider exploded" in response.json()["detail"]

    reset_settings()
    reset_engine()


def test_render_supabase_mode_does_not_load_artifact_engine(tmp_path, monkeypatch):
    repository = _seed_repository()
    artifact_root = tmp_path / "missing-artifacts"
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    monkeypatch.setenv("FORESIGHT_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FORESIGHT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("FORESIGHT_REQUIRE_SUPABASE", "true")
    monkeypatch.setenv("FORESIGHT_LOAD_ARTIFACT_ENGINE", "false")
    monkeypatch.setattr(app_main, "build_market_repository", lambda settings: repository)
    reset_settings()
    reset_engine()

    with TestClient(app_main.create_app()) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["ready"] is True
        assert health.json()["market_data"]["source"] == "memory"

        universe = client.get("/api/universe")
        assert universe.status_code == 200
        assert universe.json()["source"] == "supabase"

        forecast = client.post(
            "/api/forecasts/ticker",
            json={"ticker": "AAA", "horizon_days": 30, "window_size": 20},
        )
        assert forecast.status_code == 200
        assert forecast.json()["source"] == "supabase_ohlcv"

        models = client.get("/api/models")
        assert models.status_code == 503

    reset_settings()
    reset_engine()


def test_required_supabase_mode_reports_unhealthy_without_repository(tmp_path, monkeypatch):
    artifact_root = tmp_path / "missing-artifacts"
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    monkeypatch.setenv("FORESIGHT_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FORESIGHT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("FORESIGHT_REQUIRE_SUPABASE", "true")
    monkeypatch.setenv("FORESIGHT_LOAD_ARTIFACT_ENGINE", "false")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setattr(app_main, "build_market_repository", lambda settings: None)
    reset_settings()
    reset_engine()

    with TestClient(app_main.create_app()) as client:
        health = client.get("/api/health")
        assert health.status_code == 503
        assert "Supabase market data is required" in health.json()["detail"]

    reset_settings()
    reset_engine()


def test_required_supabase_mode_reports_unhealthy_when_schema_validation_fails(
    tmp_path, monkeypatch
):
    class BrokenRepository(InMemoryMarketDataRepository):
        def validate_schema(self):
            raise MarketDataUnavailable("missing market_index_snapshots table")

    artifact_root = tmp_path / "missing-artifacts"
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    monkeypatch.setenv("FORESIGHT_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FORESIGHT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("FORESIGHT_REQUIRE_SUPABASE", "true")
    monkeypatch.setenv("FORESIGHT_LOAD_ARTIFACT_ENGINE", "false")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test")
    monkeypatch.setattr(app_main, "build_market_repository", lambda settings: BrokenRepository())
    reset_settings()
    reset_engine()

    with TestClient(app_main.create_app()) as client:
        health = client.get("/api/health")
        assert health.status_code == 503
        assert "missing market_index_snapshots table" in health.json()["detail"]

    reset_settings()
    reset_engine()


def resettable_settings():
    reset_settings()
    return app_main.get_settings()
