from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FailureType(StrEnum):
    PERCEPTION_FAILURE = "PERCEPTION_FAILURE"
    OBJECT_LOST = "OBJECT_LOST"
    LOW_PERCEPTION_CONFIDENCE = "LOW_PERCEPTION_CONFIDENCE"
    AMBIGUOUS_IDENTITY = "AMBIGUOUS_IDENTITY"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    GRASP_FAILURE = "GRASP_FAILURE"
    OBJECT_SLIP = "OBJECT_SLIP"
    ALIGNMENT_FAILURE = "ALIGNMENT_FAILURE"
    COLLISION = "COLLISION"
    IK_FAILURE = "IK_FAILURE"
    JOINT_LIMIT = "JOINT_LIMIT"
    WORLD_MODEL_ERROR = "WORLD_MODEL_ERROR"
    CONTROL_FAILURE = "CONTROL_FAILURE"
    UNSAFE_PLAN = "UNSAFE_PLAN"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Goal:
    task: str
    object_id: str
    target_id: str
    success_conditions: dict[str, Any]


@dataclass(frozen=True)
class Observation:
    scene_id: str
    objects: dict[str, dict[str, Any]]
    targets: dict[str, dict[str, Any]]
    grasped_object: str | None = None
    sequence: int = 0


@dataclass(frozen=True)
class CandidateAction:
    action_id: str
    skill_name: str
    parameters: dict[str, Any]
    confidence: float = 1.0


@dataclass(frozen=True)
class Prediction:
    candidate: CandidateAction
    predicted_success: float
    predicted_collision: bool
    uncertainty: float
    source: str
    predicted_next_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillResult:
    success: bool
    metrics: dict[str, Any]
    failure_type: FailureType | None = None
    message: str = ""


@dataclass(frozen=True)
class RuntimeResult:
    success: bool
    actions: int
    replans: int
    episode_paths: tuple[str, ...]
    final_observation: Observation
    metrics: dict[str, Any]
