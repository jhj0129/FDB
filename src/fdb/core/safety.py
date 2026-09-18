from __future__ import annotations

import math

from .models import Observation, Prediction, SafetyDecision


class DeterministicSafetyGate:
    """Non-bypassable final gate independent of learned confidence."""

    def check(self, observation: Observation, prediction: Prediction) -> SafetyDecision:
        parameters = prediction.candidate.parameters
        reasons: list[str] = []
        numeric = [value for value in parameters.values() if isinstance(value, (int, float))]
        if any(not math.isfinite(float(value)) for value in numeric):
            reasons.append("INVALID_NUMERIC_STATE")
        if parameters.get("clearance_m", 0.0) < 0.08:
            reasons.append("INSUFFICIENT_TABLE_CLEARANCE")
        object_id = parameters.get("object_id")
        target_id = parameters.get("target_id")
        if object_id not in observation.objects or target_id not in observation.targets:
            reasons.append("UNKNOWN_SCENE_ENTITY")
        elif observation.objects[object_id]["shape"] != observation.targets[target_id]["shape"]:
            reasons.append("SHAPE_MISMATCH")
        if prediction.predicted_collision:
            reasons.append("PREDICTED_COLLISION")
        return SafetyDecision(not reasons, tuple(reasons))
