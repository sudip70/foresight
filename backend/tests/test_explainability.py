from fastapi.testclient import TestClient

from backend.app.core.config import reset_settings
from backend.app.main import create_app
from backend.app.ml.pipeline import reset_engine
from backend.tests.helpers import build_fixture_artifact_tree


def test_explanations_return_unavailable_when_threshold_is_too_high(tmp_path, monkeypatch):
    artifact_root = build_fixture_artifact_tree(tmp_path)
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("FORESIGHT_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FORESIGHT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("FORESIGHT_SURROGATE_SAMPLE_SIZE", "24")
    monkeypatch.setenv("FORESIGHT_SURROGATE_FIDELITY_THRESHOLD", "1.1")

    reset_settings()
    reset_engine()

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/explanations",
            json={"amount": 10000, "risk": 0.4, "duration": 30, "window_size": 5},
        )
        assert response.status_code == 200
        payload = response.json()
        assert any(not target["available"] for target in payload["targets"])
