from __future__ import annotations

from dataclasses import dataclass


GROUP_ORDER = (
    "expected_returns",
    "covariance_risk",
    "previous_weights",
    "sub_agent_signals",
    "class_context",
    "portfolio_state",
    "micro_indicators",
    "macro_indicators",
    "regime_features",
)


@dataclass(frozen=True)
class FeatureSlices:
    expected_returns: slice
    covariance_risk: slice
    previous_weights: slice
    sub_agent_signals: slice
    class_context: slice
    portfolio_state: slice
    micro_indicators: slice
    macro_indicators: slice
    regime_features: slice

    def as_dict(self) -> dict[str, slice]:
        return {
            "expected_returns": self.expected_returns,
            "covariance_risk": self.covariance_risk,
            "previous_weights": self.previous_weights,
            "sub_agent_signals": self.sub_agent_signals,
            "class_context": self.class_context,
            "portfolio_state": self.portfolio_state,
            "micro_indicators": self.micro_indicators,
            "macro_indicators": self.macro_indicators,
            "regime_features": self.regime_features,
        }


def build_feature_slices(
    *,
    n_assets: int,
    micro_dim: int,
    macro_dim: int,
    class_feature_dim: int = 9,
    regime_dim: int = 3,
) -> FeatureSlices:
    expected_returns = slice(0, n_assets)
    covariance_risk = slice(expected_returns.stop, expected_returns.stop + n_assets)
    previous_weights = slice(covariance_risk.stop, covariance_risk.stop + n_assets)
    sub_agent_signals = slice(previous_weights.stop, previous_weights.stop + n_assets)
    class_context = slice(sub_agent_signals.stop, sub_agent_signals.stop + class_feature_dim)
    portfolio_state = slice(class_context.stop, class_context.stop + 2)
    micro_indicators = slice(portfolio_state.stop, portfolio_state.stop + micro_dim)
    macro_indicators = slice(micro_indicators.stop, micro_indicators.stop + macro_dim)
    regime_features = slice(macro_indicators.stop, macro_indicators.stop + regime_dim)
    return FeatureSlices(
        expected_returns=expected_returns,
        covariance_risk=covariance_risk,
        previous_weights=previous_weights,
        sub_agent_signals=sub_agent_signals,
        class_context=class_context,
        portfolio_state=portfolio_state,
        micro_indicators=micro_indicators,
        macro_indicators=macro_indicators,
        regime_features=regime_features,
    )


def group_feature_values(values, feature_slices: FeatureSlices) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for name, data_slice in feature_slices.as_dict().items():
        grouped[name] = float(values[data_slice].sum())
    return grouped
