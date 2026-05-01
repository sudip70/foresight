from __future__ import annotations


class ArtifactValidationError(ValueError):
    """Raised when model artifacts or ticker inputs fail validation."""


class ExplainabilityUnavailable(RuntimeError):
    """Raised when explainability tooling is unavailable for the current environment."""
