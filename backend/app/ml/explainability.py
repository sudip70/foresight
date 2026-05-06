from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.app.ml.errors import ExplainabilityUnavailable
from backend.app.ml.feature_groups import group_feature_values
from backend.app.ml.pipeline import ForesightEngine, InferenceResult


@dataclass
class TargetExplanation:
    target: str
    available: bool
    fidelity: float
    grouped_contributions: dict[str, float]
    top_positive_drivers: list[dict]
    top_negative_drivers: list[dict]
    plain_language: str | None


def _build_target_map(result: InferenceResult) -> dict[str, np.ndarray]:
    by_class = {
        allocation["asset_class"]: allocation["weight"]
        for allocation in result.class_allocations
    }
    by_asset = {allocation["ticker"]: allocation["weight"] for allocation in result.asset_allocations}
    target_map = {
        "stock_allocation_total": np.array([by_class["stock"]], dtype=float),
        "crypto_allocation_total": np.array([by_class["crypto"]], dtype=float),
        "etf_allocation_total": np.array([by_class["etf"]], dtype=float),
    }
    for ticker in result.top_asset_targets:
        target_map[f"asset_weight:{ticker}"] = np.array([by_asset[ticker]], dtype=float)
    return target_map


def _plain_language_summary(target: str, grouped_contributions: dict[str, float]) -> str:
    ordered = sorted(
        grouped_contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    if not ordered:
        return f"No dominant drivers were identified for {target}."

    dominant_group, dominant_value = ordered[0]
    direction = "increased" if dominant_value >= 0 else "reduced"
    if len(ordered) == 1:
        return f"{target} was mainly {direction} by {dominant_group}."

    secondary_group, secondary_value = ordered[1]
    secondary_direction = "reinforced" if secondary_value >= 0 else "offset"
    return (
        f"{target} was mainly {direction} by {dominant_group}, "
        f"while {secondary_group} {secondary_direction} that signal."
    )


def _to_driver_list(grouped_contributions: dict[str, float], reverse: bool) -> list[dict]:
    ordered = sorted(
        grouped_contributions.items(),
        key=lambda item: item[1],
        reverse=reverse,
    )
    return [{"group": name, "value": float(value)} for name, value in ordered[:3]]


def build_explanations(
    *,
    engine: ForesightEngine,
    inference_result: InferenceResult,
    amount: float,
    risk: float,
    duration: int,
    window_size: int,
    requested_targets: list[str] | None,
) -> list[TargetExplanation]:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
    except ImportError as exc:  # pragma: no cover - slim deployment guard
        raise ExplainabilityUnavailable("scikit-learn is not installed") from exc

    try:
        import shap
    except ImportError as exc:  # pragma: no cover - covered by health endpoint
        raise ExplainabilityUnavailable("shap is not installed") from exc

    aligned_rows = engine._combined_context["aligned_rows"]
    start_step = min(window_size, aligned_rows - 2)
    candidate_steps = np.arange(start_step, aligned_rows - 1)
    if candidate_steps.size == 0:
        raise ExplainabilityUnavailable("Not enough data to build surrogate explanations")

    if candidate_steps.size > engine.settings.surrogate_sample_size:
        sample_index = np.linspace(
            0,
            candidate_steps.size - 1,
            num=engine.settings.surrogate_sample_size,
            dtype=int,
        )
        candidate_steps = candidate_steps[sample_index]

    current_targets = _build_target_map(inference_result)
    target_names = list(current_targets.keys())
    if requested_targets:
        target_names = [name for name in target_names if name in requested_targets]

    X = []
    y = {name: [] for name in target_names}
    for step in candidate_steps:
        scenario = engine._build_step_scenario(
            step=int(step),
            risk=risk,
            window_size=window_size,
            prev_weights=None,
        )
        raw_weights, _ = engine._predict_mixed_policy_weights(
            scenario,
            risk=risk,
            duration=duration,
        )
        weights, _ = engine._apply_risk_adjustments(
            raw_weights,
            risk,
            mu_all=scenario["mu_all"],
            cov_diag=np.diag(scenario["cov_all"]),
            cash_prior=(
                None
                if engine._cash_index() is None
                else float(scenario["sub_agent_weights"][engine._cash_index()])
            ),
        )
        X.append(scenario["meta_observation"])
        class_ranges = engine._class_ranges()
        target_values = {
            "stock_allocation_total": float(weights[class_ranges["stock"][0] : class_ranges["stock"][1]].sum()),
            "crypto_allocation_total": float(weights[class_ranges["crypto"][0] : class_ranges["crypto"][1]].sum()),
            "etf_allocation_total": float(weights[class_ranges["etf"][0] : class_ranges["etf"][1]].sum()),
        }
        asset_lookup = {
            allocation["ticker"]: float(weight)
            for allocation, weight in zip(inference_result.asset_allocations, weights)
        }
        for ticker in inference_result.top_asset_targets:
            target_values[f"asset_weight:{ticker}"] = asset_lookup[ticker]

        for name in target_names:
            y[name].append(target_values[name])

    X_array = np.asarray(X, dtype=float)
    current_observation = np.asarray(inference_result.latest_observation, dtype=float).reshape(1, -1)
    explanations: list[TargetExplanation] = []

    for target_name in target_names:
        y_array = np.asarray(y[target_name], dtype=float)
        if len(np.unique(y_array)) <= 1:
            grouped = {
                name: 0.0 for name in inference_result.feature_slices.as_dict().keys()
            }
            explanations.append(
                TargetExplanation(
                    target=target_name,
                    available=True,
                    fidelity=1.0,
                    grouped_contributions=grouped,
                    top_positive_drivers=[],
                    top_negative_drivers=[],
                    plain_language=f"{target_name} is fixed for the current fixture policy.",
                )
            )
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        fidelity = float(model.score(X_test, y_test)) if len(y_test) > 0 else 0.0

        if fidelity < engine.settings.surrogate_fidelity_threshold:
            explanations.append(
                TargetExplanation(
                    target=target_name,
                    available=False,
                    fidelity=fidelity,
                    grouped_contributions={},
                    top_positive_drivers=[],
                    top_negative_drivers=[],
                    plain_language=None,
                )
            )
            continue

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(current_observation)
        shap_vector = np.asarray(shap_values, dtype=float).reshape(-1)
        grouped = group_feature_values(shap_vector, inference_result.feature_slices)

        explanations.append(
            TargetExplanation(
                target=target_name,
                available=True,
                fidelity=fidelity,
                grouped_contributions=grouped,
                top_positive_drivers=_to_driver_list(grouped, reverse=True),
                top_negative_drivers=_to_driver_list(grouped, reverse=False),
                plain_language=_plain_language_summary(target_name, grouped),
            )
        )

    return explanations
