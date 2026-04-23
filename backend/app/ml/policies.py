from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from gymnasium import spaces
import numpy as np

from backend.app.ml.numpy_compat import install_numpy_pickle_compat


class PolicyLoadError(RuntimeError):
    """Raised when a policy bundle cannot be loaded."""


def normalize_weights(weights: np.ndarray, *, max_weight: float | None = None) -> np.ndarray:
    clipped = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, None)
    total = clipped.sum()
    if total <= 0:
        normalized = np.ones_like(clipped) / len(clipped)
    else:
        normalized = clipped / total

    if max_weight is None:
        return normalized

    cap = max(float(max_weight), 1.0 / len(normalized))
    capped = normalized.copy()
    for _ in range(len(capped) + 1):
        over_cap = capped > cap
        if not over_cap.any():
            return capped / capped.sum()

        excess = float((capped[over_cap] - cap).sum())
        capped[over_cap] = cap
        under_cap = ~over_cap
        capacity = cap - capped[under_cap]
        if not under_cap.any() or capacity.sum() <= 1e-12:
            return capped / capped.sum()

        redistributor = capped[under_cap].copy()
        if redistributor.sum() <= 1e-12:
            redistributor = capacity
        capped[under_cap] += excess * (redistributor / redistributor.sum())

    return capped / capped.sum()


def apply_class_guardrails(
    weights: np.ndarray,
    *,
    class_ranges: dict[str, tuple[int, int]],
    max_class_weights: dict[str, float] | None = None,
    max_asset_weight: float | None = None,
) -> np.ndarray:
    adjusted = normalize_weights(weights, max_weight=max_asset_weight)
    if not max_class_weights:
        return adjusted

    capped_classes: set[str] = set()
    for _ in range(len(max_class_weights) + 1):
        excess = 0.0
        for asset_class, max_weight in max_class_weights.items():
            if asset_class not in class_ranges:
                continue
            start, end = class_ranges[asset_class]
            class_weight = float(adjusted[start:end].sum())
            cap = float(max_weight)
            if class_weight > cap:
                if class_weight > 1e-12:
                    adjusted[start:end] *= cap / class_weight
                excess += class_weight - cap
                capped_classes.add(asset_class)

        if excess <= 1e-12:
            return normalize_weights(adjusted, max_weight=max_asset_weight)

        eligible_indices: list[int] = []
        capacities: list[float] = []
        for asset_class, (start, end) in class_ranges.items():
            if asset_class in capped_classes:
                continue
            class_cap = max_class_weights.get(asset_class, 1.0)
            class_weight = float(adjusted[start:end].sum())
            class_capacity = max(float(class_cap) - class_weight, 0.0)
            if class_capacity <= 1e-12:
                continue
            width = end - start
            eligible_indices.extend(range(start, end))
            capacities.extend([class_capacity / max(width, 1)] * width)

        if not eligible_indices or sum(capacities) <= 1e-12:
            return normalize_weights(adjusted, max_weight=max_asset_weight)

        indices = np.asarray(eligible_indices, dtype=int)
        capacity = np.asarray(capacities, dtype=float)
        current = adjusted[indices]
        basis = current if current.sum() > 1e-12 else capacity
        allocation = excess * (basis / basis.sum())
        adjusted[indices] += np.minimum(allocation, capacity)
        adjusted = normalize_weights(adjusted, max_weight=max_asset_weight)

    return normalize_weights(adjusted, max_weight=max_asset_weight)


def align_observation(observation: np.ndarray, target_dim: int | None) -> np.ndarray:
    vector = np.asarray(observation, dtype=np.float32).reshape(-1)
    if target_dim is None or target_dim == vector.shape[0]:
        return vector
    if vector.shape[0] < target_dim:
        return np.concatenate(
            [vector, np.zeros(target_dim - vector.shape[0], dtype=np.float32)]
        )
    return vector[:target_dim]


@dataclass
class FixedWeightPolicy:
    weights: np.ndarray
    observation_dim: int | None = None

    def predict(self, observation: np.ndarray) -> np.ndarray:
        align_observation(observation, self.observation_dim)
        return normalize_weights(self.weights)


@dataclass
class LinearPolicy:
    weights_matrix: np.ndarray
    bias: np.ndarray
    observation_dim: int

    def predict(self, observation: np.ndarray) -> np.ndarray:
        aligned = align_observation(observation, self.observation_dim)
        raw = self.weights_matrix @ aligned + self.bias
        return normalize_weights(raw)


@dataclass
class SB3Policy:
    model: object
    observation_dim: int

    def predict(self, observation: np.ndarray) -> np.ndarray:
        aligned = align_observation(observation, self.observation_dim).reshape(1, -1)
        action, _ = self.model.predict(aligned, deterministic=True)
        return normalize_weights(np.asarray(action, dtype=float))


def _build_box_spaces(
    *, observation_dim: int | None, action_dim: int | None
) -> tuple[spaces.Box, spaces.Box] | tuple[None, None]:
    if observation_dim is None or action_dim is None:
        return None, None
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(int(observation_dim),),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(int(action_dim),),
        dtype=np.float32,
    )
    return observation_space, action_space


def _load_sb3_policy(
    model_path: Path,
    algorithm: str,
    *,
    observation_dim: int | None = None,
    action_dim: int | None = None,
) -> SB3Policy:
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:  # pragma: no cover - exercised by health checks
        raise PolicyLoadError("stable-baselines3 is not installed") from exc

    install_numpy_pickle_compat()
    observation_space, action_space = _build_box_spaces(
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
    custom_objects = None
    if observation_space is not None and action_space is not None:
        custom_objects = {
            "observation_space": observation_space,
            "action_space": action_space,
        }

    algorithm = algorithm.lower()
    if algorithm == "ppo":
        model = PPO.load(model_path, custom_objects=custom_objects)
    elif algorithm == "sac":
        model = SAC.load(model_path, custom_objects=custom_objects)
    else:  # pragma: no cover - configuration error
        raise PolicyLoadError(f"Unsupported SB3 algorithm: {algorithm}")

    observation_dim = int(np.prod(model.observation_space.shape))
    return SB3Policy(model=model, observation_dim=observation_dim)


def load_policy(
    model_path: Path,
    metadata: dict,
    *,
    observation_dim: int | None = None,
    action_dim: int | None = None,
) -> FixedWeightPolicy | SB3Policy:
    backend = metadata.get("policy_backend", "sb3").lower()
    if backend == "fixed":
        payload = json.loads(model_path.read_text())
        return FixedWeightPolicy(
            weights=np.asarray(payload["weights"], dtype=float),
            observation_dim=payload.get("observation_dim"),
        )
    if backend == "linear":
        payload = json.loads(model_path.read_text())
        matrix = np.asarray(payload["weights_matrix"], dtype=float)
        bias = payload.get("bias")
        return LinearPolicy(
            weights_matrix=matrix,
            bias=np.asarray(
                bias if bias is not None else np.zeros(matrix.shape[0], dtype=float),
                dtype=float,
            ),
            observation_dim=int(payload["observation_dim"]),
        )

    return _load_sb3_policy(
        model_path,
        metadata.get("algorithm", "ppo"),
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
