from __future__ import annotations

from .models import CandidateAction, Observation, Prediction


class HybridShapeWorldModel:
    """Fast neural-surrogate interface with an explicit non-physics fallback.

    The fallback is deterministic geometry for unit tests. It is deliberately not
    counted as MuJoCo verification; physical adapters must report that separately.
    """

    def __init__(self, *, neural_uncertainty_threshold: float = 0.20) -> None:
        self.neural_uncertainty_threshold = neural_uncertainty_threshold
        self.calls = 0
        self.physics_fallbacks = 0
        self.deterministic_fallbacks = 0

    def predict(self, observation: Observation, candidate: CandidateAction) -> Prediction:
        self.calls += 1
        bias = abs(float(candidate.parameters.get("rotation_bias_deg", 0.0)))
        uncertainty = min(1.0, bias / 45.0)
        source = "neural"
        if uncertainty > self.neural_uncertainty_threshold:
            source = "deterministic_geometry_fallback"
            self.deterministic_fallbacks += 1
        probability = max(0.01, 1.0 - bias / 30.0)
        return Prediction(
            candidate=candidate,
            predicted_success=probability,
            predicted_collision=False,
            uncertainty=uncertainty,
            source=source,
            predicted_next_state={"inside": bias <= 8.0, "released": bias <= 8.0},
        )
