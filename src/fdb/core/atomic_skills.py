from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .models import CandidateAction, FailureType, Goal, Observation


Predicate = Callable[[dict[str, Any], dict[str, Any]], bool]


@dataclass(frozen=True)
class AtomicSkill:
    name: str
    capability: str
    preconditions: tuple[str, ...]
    effects: dict[str, Any]
    failures: tuple[FailureType, ...]
    predicate: Predicate
    provenance: tuple[str, ...] = ("src/fdb/core/atomic_skills.py",)

    def rejection_reasons(self, observation: Observation, goal: Goal) -> tuple[str, ...]:
        if goal.object_id not in observation.objects:
            return ("OBJECT_NOT_OBSERVED",)
        if goal.target_id not in observation.targets:
            return ("TARGET_NOT_OBSERVED",)
        obj = observation.objects[goal.object_id]
        target = observation.targets[goal.target_id]
        return () if self.predicate(obj, target) else ("PRECONDITIONS_NOT_MET",)

    def applicable(self, observation: Observation, goal: Goal) -> bool:
        return not self.rejection_reasons(observation, goal)

    def propose(
        self, observation: Observation, goal: Goal, failure: FailureType | None,
    ) -> list[CandidateAction]:
        obj = observation.objects[goal.object_id]
        target = observation.targets[goal.target_id]
        common = {
            "object_id": goal.object_id,
            "target_id": goal.target_id,
            "clearance_m": 0.10,
            "object_pose": obj.get("pose"),
            "target_pose": target.get("pose"),
            "object_yaw": obj.get("orientation", 0.0),
            "target_yaw": target.get("orientation", 0.0),
        }
        if self.name in {"align_object", "recover_alignment"}:
            yaw_error = float(obj.get("yaw_error_deg", 0.0))
            offsets = (0.0, -5.0, 5.0) if self.name == "recover_alignment" else (0.0,)
            return [CandidateAction(
                f"{self.name}:{goal.object_id}:{offset:+.1f}", self.name,
                {**common, "rotation_correction_deg": -yaw_error + offset}, 0.9,
            ) for offset in offsets]
        return [CandidateAction(
            f"{self.name}:{goal.object_id}", self.name, common, 0.95,
        )]


def build_shape_manipulation_skills() -> tuple[AtomicSkill, ...]:
    """State-machine skill library; order has no execution meaning."""
    return (
        AtomicSkill(
            "observe_scene", "perception", ("object or target not visible",),
            {"visible": True, "target_visible": True}, (FailureType.PERCEPTION_FAILURE,),
            lambda obj, target: not obj.get("visible", False) or not target.get("visible", False),
        ),
        AtomicSkill(
            "reach_object", "manipulation", ("visible", "not grasped", "not ee_near"),
            {"ee_near": True}, (FailureType.IK_FAILURE, FailureType.COLLISION),
            lambda obj, target: obj.get("visible", False) and not obj.get("grasped", False)
            and not obj.get("ee_near", False) and not obj.get("inserted", False),
        ),
        AtomicSkill(
            "grasp_object", "manipulation", ("ee_near", "gripper open"),
            {"grasped": True}, (FailureType.GRASP_FAILURE, FailureType.OBJECT_SLIP),
            lambda obj, target: obj.get("ee_near", False) and not obj.get("grasped", False)
            and not obj.get("inserted", False),
        ),
        AtomicSkill(
            "lift_object", "manipulation", ("grasped", "not lifted"),
            {"lifted": True}, (FailureType.OBJECT_SLIP, FailureType.COLLISION),
            lambda obj, target: obj.get("grasped", False) and not obj.get("lifted", False),
        ),
        AtomicSkill(
            "align_object", "manipulation", ("grasped", "lifted", "yaw error"),
            {"aligned": True}, (FailureType.ALIGNMENT_FAILURE,),
            lambda obj, target: obj.get("grasped", False) and obj.get("lifted", False)
            and not obj.get("aligned", False) and obj.get("last_failure") != FailureType.ALIGNMENT_FAILURE,
        ),
        AtomicSkill(
            "recover_alignment", "recovery", ("alignment failure", "object retained"),
            {"aligned": True}, (FailureType.ALIGNMENT_FAILURE,),
            lambda obj, target: obj.get("grasped", False) and obj.get("lifted", False)
            and obj.get("last_failure") == FailureType.ALIGNMENT_FAILURE,
        ),
        AtomicSkill(
            "move_to_target", "manipulation", ("grasped", "lifted", "aligned"),
            {"near_target": True}, (FailureType.IK_FAILURE, FailureType.COLLISION),
            lambda obj, target: obj.get("grasped", False) and obj.get("lifted", False)
            and obj.get("aligned", False) and not obj.get("near_target", False),
        ),
        AtomicSkill(
            "insert_object", "manipulation", ("near target", "aligned", "grasped"),
            {"inserted": True}, (FailureType.ALIGNMENT_FAILURE, FailureType.COLLISION),
            lambda obj, target: obj.get("near_target", False) and obj.get("aligned", False)
            and obj.get("grasped", False) and not obj.get("inserted", False),
        ),
        AtomicSkill(
            "release_object", "manipulation", ("inserted", "grasped"),
            {"released": True, "grasped": False}, (FailureType.OBJECT_SLIP,),
            lambda obj, target: obj.get("inserted", False) and obj.get("grasped", False)
            and not obj.get("released", False),
        ),
        AtomicSkill(
            "retreat", "manipulation", ("inserted", "released"),
            {"retreated": True}, (FailureType.IK_FAILURE, FailureType.COLLISION),
            lambda obj, target: obj.get("inserted", False) and obj.get("released", False)
            and not obj.get("retreated", False),
        ),
    )


def infer_subgoal(observation: Observation, goal: Goal) -> str:
    if goal.object_id not in observation.objects:
        return "localize_object"
    obj = observation.objects[goal.object_id]
    if not obj.get("visible"):
        return "localize_object"
    if obj.get("inserted"):
        if not obj.get("released"):
            return "release_object"
        if goal.success_conditions.get("retreated") and not obj.get("retreated"):
            return "retreat"
        return "goal_satisfied"
    if not obj.get("ee_near") and not obj.get("grasped"):
        return "reach_object"
    if not obj.get("grasped"):
        return "grasp_object"
    if not obj.get("lifted"):
        return "lift_object"
    if not obj.get("aligned"):
        return "recover_alignment" if obj.get("last_failure") else "align_object"
    if not obj.get("near_target"):
        return "move_to_target"
    return "insert_object"


def generate_skill_candidates(
    skills: tuple[AtomicSkill, ...], observation: Observation, goal: Goal,
    failure: FailureType | None = None,
) -> tuple[list[CandidateAction], dict[str, tuple[str, ...]]]:
    candidates: list[CandidateAction] = []
    rejected: dict[str, tuple[str, ...]] = {}
    for skill in skills:
        reasons = skill.rejection_reasons(observation, goal)
        if reasons:
            rejected[skill.name] = reasons
        else:
            candidates.extend(skill.propose(observation, goal, failure))
    return candidates, rejected
