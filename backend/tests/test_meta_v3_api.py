from fastapi.testclient import TestClient

from backend.app.core.config import reset_settings
from backend.app.main import create_app
from backend.app.ml.pipeline import reset_engine
from backend.tests.helpers import build_fixture_artifact_tree


def test_v3_models_endpoint_reports_shared_macro_architecture(tmp_path, monkeypatch):
    artifact_root = build_fixture_artifact_tree(tmp_path, version="v3")
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("FORESIGHT_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FORESIGHT_DATASET_ROOT", str(dataset_root))

    reset_settings()
    reset_engine()

    with TestClient(create_app()) as client:
        response = client.get("/api/models")
        assert response.status_code == 200
        payload = response.json()
        assert payload["meta_agent"]["feature_version"] == "sac-meta-v3-globalmacro-cashaware"
        assert payload["meta_agent"]["uses_shared_macro"] is True
        assert payload["meta_agent"]["class_feature_dim"] == 12
        assert payload["meta_agent"]["sub_agent_consensus_blend"] > 0
        assert payload["feature_groups"]["class_context"]["stop"] - payload["feature_groups"]["class_context"]["start"] == 12


def test_v3_inference_uses_global_macro_snapshot_and_cash_aware_sub_agents(tmp_path, monkeypatch):
    artifact_root = build_fixture_artifact_tree(tmp_path, version="v3")
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("FORESIGHT_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FORESIGHT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("FORESIGHT_TOP_ASSET_TARGET_COUNT", "2")

    reset_settings()
    reset_engine()

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/inference",
            json={"amount": 12000, "risk": 0.5, "duration": 30, "window_size": 5},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["model_version"] == "sac-meta-v3-globalmacro-cashaware"
        assert len(payload["asset_allocations"]) == 7
        assert payload["latest_snapshot"]["macro"][0]["name"] == "vix_market_volatility"
        assert payload["latest_snapshot"]["sub_agent_weight_sums"]["crypto"] < 1.0
        assert payload["latest_snapshot"]["policy_mix"]["sub_agent_consensus_blend"] > 0
