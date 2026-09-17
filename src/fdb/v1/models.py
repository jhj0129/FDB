from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PushCandidate:
    plan_id: str
    force_newtons: float
    force_steps: int
    settle_steps: int
    rationale: str


@dataclass(frozen=True)
class FeedbackCandidate:
    plan_id: str
    kp: float
    kd: float
    max_force_newtons: float
    control_steps: int
    settle_steps: int
    rationale: str


@dataclass(frozen=True)
class PhysicsScenario:
    scenario_id: str = "nominal"
    mass_scale: float = 1.0
    sliding_friction: float = 0.8
    initial_offset_x: float = 0.0
    initial_offset_y: float = 0.0


@dataclass(frozen=True)
class PhysicsEvaluation:
    success: bool
    score: float
    final_distance: float
    path_length: float
    execution_time: float
    unintended_contact_count: int
    out_of_bounds: bool
    stable: bool


@dataclass(frozen=True)
class PhysicsRollout:
    plan: PushCandidate | FeedbackCandidate
    initial_position: tuple[float, float]
    target_position: tuple[float, float]
    final_position: tuple[float, float]
    evaluation: PhysicsEvaluation
