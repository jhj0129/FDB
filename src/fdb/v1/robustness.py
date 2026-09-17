from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from statistics import mean

from .environment import MujocoPushEnvironment
from .models import FeedbackCandidate, PhysicsRollout, PhysicsScenario
from .planner import PushCandidatePlanner


@dataclass(frozen=True)
class MethodSummary:
    method: str
    success_count: int
    trial_count: int
    success_rate: float
    mean_final_distance: float
    mean_score: float


@dataclass(frozen=True)
class RobustnessResult:
    scenarios: tuple[PhysicsScenario, ...]
    open_loop_rollouts: tuple[PhysicsRollout, ...]
    feedback_rollouts: tuple[PhysicsRollout, ...]
    open_loop_summary: MethodSummary
    feedback_summary: MethodSummary

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def generate_scenarios(count: int = 30, seed: int = 1701) -> tuple[PhysicsScenario, ...]:
    generator = random.Random(seed)
    return tuple(
        PhysicsScenario(
            scenario_id=f"scenario_{index:03d}",
            mass_scale=generator.uniform(0.4, 1.8),
            sliding_friction=generator.uniform(0.2, 1.4),
            initial_offset_x=generator.uniform(-0.15, 0.15),
            initial_offset_y=generator.uniform(-0.15, 0.15),
        )
        for index in range(count)
    )


def _summarize(method: str, rollouts: tuple[PhysicsRollout, ...]) -> MethodSummary:
    successes = sum(item.evaluation.success for item in rollouts)
    return MethodSummary(
        method=method,
        success_count=successes,
        trial_count=len(rollouts),
        success_rate=successes / len(rollouts),
        mean_final_distance=mean(item.evaluation.final_distance for item in rollouts),
        mean_score=mean(item.evaluation.score for item in rollouts),
    )


class RobustnessExperiment:
    def __init__(self) -> None:
        self.environment = MujocoPushEnvironment()
        self.open_loop_plan = PushCandidatePlanner().create_candidates()[1]
        self.feedback_plan = FeedbackCandidate(
            plan_id="pd_feedback",
            kp=18.0,
            kd=5.0,
            max_force_newtons=8.0,
            control_steps=700,
            settle_steps=300,
            rationale="Closed-loop PD force corrects position and velocity every step.",
        )

    def run(self, scenarios: tuple[PhysicsScenario, ...] | None = None) -> RobustnessResult:
        scenarios = scenarios or generate_scenarios()
        open_loop = tuple(
            self.environment.simulate(self.open_loop_plan, scenario) for scenario in scenarios
        )
        feedback = tuple(
            self.environment.simulate_feedback(self.feedback_plan, scenario)
            for scenario in scenarios
        )
        return RobustnessResult(
            scenarios=scenarios,
            open_loop_rollouts=open_loop,
            feedback_rollouts=feedback,
            open_loop_summary=_summarize("open_loop", open_loop),
            feedback_summary=_summarize("feedback", feedback),
        )
