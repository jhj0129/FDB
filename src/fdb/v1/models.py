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
    plan: PushCandidate
    initial_position: tuple[float, float]
    target_position: tuple[float, float]
    final_position: tuple[float, float]
    evaluation: PhysicsEvaluation

