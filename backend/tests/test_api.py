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


def test_inference_endpoint_returns_allocations_and_summary(client):
    response = client.post(
        "/api/inference",
        json={"amount": 15000, "risk": 0.42, "duration": 30, "window_size": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == "fixture-v1"
    assert len(payload["class_allocations"]) == 3
    assert len(payload["asset_allocations"]) == 6
    assert payload["summary"]["portfolio_variance"] >= 0


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

