from backend.app.ml.feature_groups import build_feature_slices


def test_feature_group_slices_cover_expected_ranges():
    slices = build_feature_slices(n_assets=6, micro_dim=12, macro_dim=6)
    assert slices.expected_returns == slice(0, 6)
    assert slices.covariance_risk == slice(6, 12)
    assert slices.previous_weights == slice(12, 18)
    assert slices.sub_agent_signals == slice(18, 24)
    assert slices.class_context == slice(24, 33)
    assert slices.portfolio_state == slice(33, 35)
    assert slices.micro_indicators == slice(35, 47)
    assert slices.macro_indicators == slice(47, 53)
    assert slices.regime_features == slice(53, 56)
