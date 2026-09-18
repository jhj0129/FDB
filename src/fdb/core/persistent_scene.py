from __future__ import annotations

from copy import deepcopy

from .models import CandidateAction, FailureType, Observation, SkillResult


class PersistentShapeEnvironment:
    """One non-resetting scene containing four stable object and target identities.

    This fast environment tests decision semantics. Its action contract is intentionally
    compatible with a future MuJoCo persistent-scene adapter; it does not claim to be
    physical evidence.
    """

    SHAPES = ("square", "circle", "triangle", "rectangle")

    def __init__(self, *, initial_bias_by_shape: dict[str, float] | None = None) -> None:
        initial_bias_by_shape = initial_bias_by_shape or {}
        self.scene_id = "persistent-shape-scene-v1"
        self.sequence = 0
        self.grasped_object: str | None = None
        self.objects = {
            f"{shape}_object": {
                "shape": shape, "pose": [0.40, index * 0.09 - 0.135, 0.33],
                "inside_target": None, "released": True, "completed": False,
                "initial_bias_deg": initial_bias_by_shape.get(shape, 0.0),
            }
            for index, shape in enumerate(self.SHAPES)
        }
        self.targets = {
            f"{shape}_receptacle": {
                "shape": shape, "pose": [0.62, index * 0.09 - 0.135, 0.29],
            }
            for index, shape in enumerate(self.SHAPES)
        }

    def observe(self) -> Observation:
        return Observation(
            scene_id=self.scene_id, objects=deepcopy(self.objects),
            targets=deepcopy(self.targets), grasped_object=self.grasped_object,
            sequence=self.sequence,
        )

    def execute(self, action: CandidateAction) -> SkillResult:
        self.sequence += 1
        object_id = str(action.parameters["object_id"])
        target_id = str(action.parameters["target_id"])
        bias = abs(float(action.parameters.get("rotation_bias_deg", 0.0)))
        if self.objects[object_id]["shape"] != self.targets[target_id]["shape"]:
            return SkillResult(False, {"shape_match": False}, FailureType.UNSAFE_PLAN)
        if bias > 8.0:
            return SkillResult(
                False,
                {"final_yaw_error_deg": bias, "frame_contact_steps": 12, "collision": False},
                FailureType.ALIGNMENT_FAILURE,
                "수용구 경계 접촉으로 삽입되지 않음",
            )
        self.objects[object_id].update(
            inside_target=target_id, released=True, completed=True, initial_bias_deg=0.0,
        )
        return SkillResult(
            True,
            {"final_yaw_error_deg": bias, "frame_contact_steps": 0,
             "robot_table_contact_steps": 0, "collision": False},
        )

    def goal_satisfied(self, object_id: str, target_id: str) -> bool:
        state = self.objects.get(object_id, {})
        return bool(
            state.get("inside_target") == target_id
            and state.get("released")
            and state.get("completed")
        )
