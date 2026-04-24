from __future__ import annotations

import numpy as np

from backend.tests.helpers import build_v3_fixture_artifact_tree
from offline.sac_training import SACMetaPortfolioEnv, SACMetaTrainingConfig, _build_context


def test_sac_meta_env_uses_sleeve_action_space(tmp_path):
    artifact_root = build_v3_fixture_artifact_tree(tmp_path, rows=80)
    config = SACMetaTrainingConfig(
        episode_length=12,
        window_size=5,
        risk_values=(0.5,),
        max_cash_weight=0.95,
    )
    context = _build_context(artifact_root, config)
    env = SACMetaPortfolioEnv(
        context=context,
        config=config,
        random_start=False,
        fixed_risk=0.5,
    )

    observation, _ = env.reset()
    next_observation, reward, terminated, truncated, info = env.step(
        np.asarray([0.30, 0.20, 0.40, 0.10], dtype=float)
    )

    assert env.action_space.shape == (4,)
    assert observation.shape == env.observation_space.shape
    assert next_observation.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert set(info["sleeve_weights"]) == {"stock", "crypto", "etf", "cash"}
    assert np.isclose(sum(info["sleeve_weights"].values()), 1.0)
    assert info["sleeve_weights"]["cash"] <= 0.29 + 1e-12
