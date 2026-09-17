from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from fdb.v1.environment import MujocoPushEnvironment
from fdb.v1.models import PhysicsRollout, PhysicsScenario, PushCandidate
from .portable import PortableDynamicsEnsemble, PortablePrediction


@dataclass(frozen=True)
class HybridCandidateEvaluation:
    plan: PushCandidate
    final_position: tuple[float, float]
    final_distance_m: float
    predicted_success: bool
    source: str
    uncertainty_m: float
    fallback_reasons: tuple[str, ...]


@dataclass(frozen=True)
class HybridDecisionResult:
    selected: HybridCandidateEvaluation
    candidates: tuple[HybridCandidateEvaluation, ...]
    verified_execution: PhysicsRollout
    neural_to_physics_error_m: float | None


class SafeHybridPushPlanner:
    """Use the neural model for screening but retain physics as the safety authority."""

    def __init__(self, model_path: Path, *, success_radius: float = 0.14) -> None:
        self.model = PortableDynamicsEnsemble(model_path)
        self.environment = MujocoPushEnvironment(success_radius=success_radius)
        self.success_radius = success_radius

    @staticmethod
    def feature_vector(
        initial: tuple[float, float],
        target: tuple[float, float],
        scenario: PhysicsScenario,
        plan: PushCandidate,
    ) -> np.ndarray:
        return np.asarray(
            [
                *initial,
                *target,
                scenario.mass_scale,
                scenario.sliding_friction,
                plan.force_newtons,
                float(plan.force_steps),
                float(plan.settle_steps),
            ],
            dtype=np.float32,
        )

    def evaluate_candidate(
        self,
        plan: PushCandidate,
        scenario: PhysicsScenario,
    ) -> HybridCandidateEvaluation:
        initial, target = self.environment.observe_initial(scenario)
        prediction: PortablePrediction = self.model.predict(
            self.feature_vector(initial, target, scenario, plan)
        )
        if prediction.use_physics_fallback:
            rollout = self.environment.simulate(plan, scenario)
            final = rollout.final_position
            source = "physics_fallback"
        else:
            final = prediction.final_position
            source = "neural_ensemble"
        distance = math.hypot(final[0] - target[0], final[1] - target[1])
        return HybridCandidateEvaluation(
            plan=plan,
            final_position=final,
            final_distance_m=distance,
            predicted_success=distance <= self.success_radius,
            source=source,
            uncertainty_m=prediction.uncertainty_m,
            fallback_reasons=prediction.fallback_reasons,
        )

    def decide(
        self,
        plans: tuple[PushCandidate, ...],
        scenario: PhysicsScenario | None = None,
    ) -> HybridDecisionResult:
        scenario = scenario or PhysicsScenario()
        candidates = tuple(self.evaluate_candidate(plan, scenario) for plan in plans)
        selected = max(
            candidates,
            key=lambda item: (item.predicted_success, -item.final_distance_m),
        )
        # The learned model may rank candidates, but execution is always checked by physics.
        verified = self.environment.simulate(selected.plan, scenario)
        error = None
        if selected.source == "neural_ensemble":
            error = math.hypot(
                selected.final_position[0] - verified.final_position[0],
                selected.final_position[1] - verified.final_position[1],
            )
        return HybridDecisionResult(selected, candidates, verified, error)
