from datetime import date

import pytest
from fastapi.testclient import TestClient

import backend.app.main as app_main
from backend.app.core.config import reset_settings
from backend.app.main import create_app
from backend.app.ml.pipeline import reset_engine
from backend.tests.helpers import build_fixture_artifact_tree


def test_health_endpoint_returns_dependency_and_artifact_status(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert set(payload["artifacts"].keys()) == {"stock", "crypto", "etf"}


def test_models_endpoint_exposes_feature_groups(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["supported_asset_classes"] == ["stock", "crypto", "etf"]
    assert "expected_returns" in payload["feature_groups"]


def test_universe_endpoint_returns_current_artifact_tickers(client):
    response = client.get("/api/universe")
    assert response.status_code == 200
    payload = response.json()
    assert payload["supported_asset_classes"] == ["stock", "crypto", "etf"]
    assert len(payload["tickers"]) == 6
    assert {entry["asset_class"] for entry in payload["tickers"]} == {
        "stock",
        "crypto",
        "etf",
    }
    assert all(entry["latest_price"] > 0 for entry in payload["tickers"])


def test_refresh_status_reports_local_artifact_freshness(client):
    response = client.get("/api/data/refresh/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is False
    assert payload["source"] == "local_artifacts"
    assert payload["asset_count"] == 6
    assert payload["latest_market_date"]
    assert "local artifact" in payload["message"]


def test_ticker_forecast_endpoint_returns_ordered_scenarios(client):
    response = client.post(
        "/api/forecasts/ticker",
        json={"ticker": "AAPL", "horizon_days": 45, "window_size": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["asset_class"] == "stock"
    assert payload["target_prices"]["bear"] < payload["target_prices"]["base"]
    assert payload["target_prices"]["base"] < payload["target_prices"]["bull"]
    assert len(payload["historical_prices"]) > 1
    assert payload["forecast_paths"]["base"][0]["price"] == payload["latest_price"]
    today = date.today().isoformat()
    assert payload["forecast_start_date"] == today
    assert payload["forecast_paths"]["base"][0]["date"] == today
    assert payload["data_as_of"] == payload["latest_date"]
    assert len(payload["forecast_paths"]["base"]) == 46
    assert len({round(point["price"], 4) for point in payload["forecast_paths"]["base"]}) > 10
    assert payload["return_estimator"]["method"] == "multi_window_shrunk"
    assert 0 <= payload["confidence"] <= 1


def test_local_profile_uses_artifact_and_config_fallbacks(client):
    response = client.get("/api/tickers/AAPL/profile")
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "local_artifacts"
    assert payload["display_name"] == "Apple"
    assert payload["fields"]["exchange"] == "NASDAQ"
    assert payload["fields"]["sector"] == "Technology"
    assert payload["fields"]["industry"] == "Consumer Electronics"
    assert payload["fields"]["country"] == "US"
    assert payload["fields"]["fifty_two_week_high"] >= payload["fields"]["last_sale"]
    assert payload["fields"]["fifty_two_week_low"] <= payload["fields"]["last_sale"]


def test_ticker_forecast_invalid_ticker_returns_clear_error(client):
    response = client.post(
        "/api/forecasts/ticker",
        json={"ticker": "NOTREAL", "horizon_days": 30},
    )
    assert response.status_code == 422
    assert "Unsupported ticker" in response.json()["detail"]


def test_market_forecast_endpoint_returns_ranked_tickers(client):
    response = client.post(
        "/api/forecasts/market",
        json={"horizon_days": 30, "risk": 0.6, "top_n": 4, "window_size": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["ranked_tickers"]) == 4
    scores = [entry["opportunity_score"] for entry in payload["ranked_tickers"]]
    assert scores == sorted(scores, reverse=True)
    assert "best_base_case" in payload["highlights"]
    assert "macro_snapshot" in payload


def test_portfolio_simulation_returns_forecast_driven_allocations(client):
    response = client.post(
        "/api/portfolio/simulations",
        json={"amount": 10000, "risk": 0.7, "horizon_days": 60, "window_size": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    total_weight = sum(allocation["weight"] for allocation in payload["asset_allocations"])
    assert abs(total_weight - 1.0) < 1e-6
    assert payload["summary"]["bear_value"] < payload["summary"]["base_value"]
    assert payload["summary"]["base_value"] < payload["summary"]["bull_value"]
    assert payload["trade_plan"]
    assert any(allocation["asset_class"] == "cash" for allocation in payload["asset_allocations"])


def test_inference_endpoint_returns_allocations_and_summary(client):
    response = client.post(
        "/api/inference",
        json={"amount": 15000, "risk": 0.42, "duration": 30, "window_size": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == "fixture-v1"
    assert len(payload["class_allocations"]) == 4
    assert len(payload["asset_allocations"]) == 7
    assert payload["summary"]["portfolio_variance"] >= 0
    assert payload["summary"]["projected_value"] > 0
    assert payload["summary"]["projected_profit"] > -15000
    assert payload["latest_snapshot"]["policy_mix"]["sub_agent_consensus_blend"] > 0
    cash_weight = next(
        allocation["weight"]
        for allocation in payload["class_allocations"]
        if allocation["asset_class"] == "cash"
    )
    assert cash_weight <= payload["latest_snapshot"]["policy_mix"]["risk_cash_cap"] + 1e-6
    assert payload["trade_log"]


def test_short_projection_window_uses_stabilized_return_estimator(client):
    response = client.post(
        "/api/inference",
        json={"amount": 10000, "risk": 0.8, "duration": 300, "window_size": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    estimator = payload["latest_snapshot"]["policy_mix"]["return_estimator"]["stock"]

    assert estimator["method"] == "multi_window_shrunk"
    assert estimator["requested_window"] == 5
    assert min(estimator["projection_windows"]) >= 20
    assert abs(payload["summary"]["projected_horizon_return"]) < 1.0


def test_explanations_endpoint_returns_grouped_contributions(client):
    response = client.post(
        "/api/explanations",
        json={"amount": 12000, "risk": 0.51, "duration": 20, "window_size": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["targets"]) >= 3
    available_targets = [target for target in payload["targets"] if target["available"]]
    assert available_targets
    assert "expected_returns" in available_targets[0]["grouped_contributions"]


def test_backtests_endpoint_returns_curves_and_metrics(client):
    response = client.post(
        "/api/backtests",
        json={"initial_amount": 10000, "risk": 0.5, "window_size": 5, "max_steps": 8},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary_metrics"]["ending_value"] > 0
    assert len(payload["equity_curve"]) == 9
    assert len(payload["drawdown_curve"]) == 9
    assert payload["trade_log"]


def test_backtest_trade_log_matches_reported_turnover(client):
    response = client.post(
        "/api/backtests",
        json={"initial_amount": 10000, "risk": 0.5, "window_size": 5, "max_steps": 8},
    )
    assert response.status_code == 200
    payload = response.json()

    logged_turnover = sum(
        trade["amount"] / trade["portfolio_value_before"]
        for trade in payload["trade_log"]
        if trade["portfolio_value_before"] > 0
    )
    reported_turnover = (
        payload["summary_metrics"]["average_daily_turnover"]
        * payload["summary_metrics"]["backtest_steps"]
    )

    assert logged_turnover == pytest.approx(reported_turnover, rel=0.15, abs=1e-3)


def test_backend_startup_refreshes_market_indices_without_supabase(tmp_path, monkeypatch):
    artifact_root = build_fixture_artifact_tree(tmp_path)
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("STOCKIFY_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("STOCKIFY_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("STOCKIFY_MARKET_INDEX_AUTO_REFRESH", "true")

    def fake_refresh(settings, *, repository=None):
        return {
            "enabled": True,
            "provider": "fake",
            "as_of_date": "2026-04-28",
            "rows_written": 0,
            "rows": [
                {
                    "symbol": "SP500",
                    "as_of_date": "2026-04-28",
                    "label": "S&P 500",
                    "display_name": "S&P 500 Index",
                    "provider_symbol": "^GSPC",
                    "value": 5100.0,
                    "previous_close": 5000.0,
                    "change": 100.0,
                    "change_percent": 0.02,
                    "currency": "USD",
                    "provider": "fake",
                    "display_order": 1,
                }
            ],
        }

    monkeypatch.setattr(app_main, "refresh_market_index_snapshots", fake_refresh)
    reset_settings()
    reset_engine()

    with TestClient(app_main.create_app()) as test_client:
        response = test_client.get("/api/market/indices")
        assert response.status_code == 200
        payload = response.json()
        assert payload["source"] == "fake"
        assert payload["as_of_date"] == "2026-04-28"
        assert payload["indices"][0]["symbol"] == "SP500"

    reset_settings()
    reset_engine()


def test_corrupted_artifacts_return_degraded_health_and_503(tmp_path, monkeypatch):
    artifact_root = build_fixture_artifact_tree(tmp_path)
    (artifact_root / "stock" / "prices.npy").unlink()
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("STOCKIFY_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("STOCKIFY_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("STOCKIFY_MARKET_INDEX_AUTO_REFRESH", "false")
    reset_settings()
    reset_engine()

    with TestClient(create_app()) as degraded_client:
        health = degraded_client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"

        response = degraded_client.get("/api/universe")
        assert response.status_code == 503

    reset_settings()
    reset_engine()
