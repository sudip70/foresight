from backend.app.core.config import get_settings, reset_settings


def test_foresight_env_vars_override_legacy_stockify_fallback(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    dataset_root = tmp_path / "datasets"
    artifact_root.mkdir()
    dataset_root.mkdir()

    monkeypatch.setenv("FORESIGHT_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FORESIGHT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("FORESIGHT_MARKET_DATA_PROVIDER", "preferred")
    monkeypatch.setenv("FORESIGHT_ARTIFACT_POLICY_MODE", "signal")
    monkeypatch.setenv("STOCKIFY_MARKET_DATA_PROVIDER", "legacy")

    reset_settings()
    settings = get_settings()

    assert settings.artifact_root == artifact_root
    assert settings.dataset_root == dataset_root
    assert settings.market_data_provider == "preferred"
    assert settings.artifact_policy_mode == "signal"

    reset_settings()


def test_legacy_stockify_env_vars_still_work(tmp_path, monkeypatch):
    artifact_root = tmp_path / "legacy-artifacts"
    dataset_root = tmp_path / "legacy-datasets"
    artifact_root.mkdir()
    dataset_root.mkdir()

    monkeypatch.delenv("FORESIGHT_ARTIFACT_ROOT", raising=False)
    monkeypatch.delenv("FORESIGHT_DATASET_ROOT", raising=False)
    monkeypatch.delenv("FORESIGHT_LOAD_ARTIFACT_ENGINE", raising=False)
    monkeypatch.setenv("STOCKIFY_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("STOCKIFY_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("STOCKIFY_LOAD_ARTIFACT_ENGINE", "false")

    reset_settings()
    settings = get_settings()

    assert settings.artifact_root == artifact_root
    assert settings.dataset_root == dataset_root
    assert settings.load_artifact_engine is False

    reset_settings()
