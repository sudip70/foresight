from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import reset_settings
from backend.app.main import create_app
from backend.app.ml.pipeline import reset_engine
from backend.tests.helpers import build_fixture_artifact_tree


@pytest.fixture()
def client(tmp_path, monkeypatch):
    artifact_root = build_fixture_artifact_tree(tmp_path)
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("FORESIGHT_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FORESIGHT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("FORESIGHT_SURROGATE_SAMPLE_SIZE", "24")
    monkeypatch.setenv("FORESIGHT_SURROGATE_FIDELITY_THRESHOLD", "0.1")
    monkeypatch.setenv("FORESIGHT_TOP_ASSET_TARGET_COUNT", "2")
    monkeypatch.setenv("FORESIGHT_DEFAULT_BACKTEST_STEPS", "12")
    monkeypatch.setenv("FORESIGHT_MARKET_INDEX_AUTO_REFRESH", "false")

    reset_settings()
    reset_engine()

    with TestClient(create_app()) as test_client:
        yield test_client

    reset_settings()
    reset_engine()
