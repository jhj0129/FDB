from __future__ import annotations

from typing import Any


MODEL_NAME = "franka_emika_panda"
MODEL_RELEASE = "2026.9.0"


def load_robot() -> tuple[Any, Any]:
    try:
        import mujoco_menagerie as menagerie
    except ImportError as error:
        raise RuntimeError(
            "FDB v2 requires: python -m pip install -e '.[robot]'"
        ) from error
    record = menagerie.get(MODEL_NAME)
    return record, record.model(record.default_model)


def inspect_robot() -> dict[str, Any]:
    import mujoco

    record, model = load_robot()
    joints: list[dict[str, Any]] = []
    for index in range(model.njnt):
        joint = model.joint(index)
        body_id = int(model.jnt_bodyid[index])
        body_name = model.body(body_id).name
        parent_id = int(model.body_parentid[body_id])
        joint_type = mujoco.mjtJoint(int(model.jnt_type[index])).name.removeprefix("mjJNT_").lower()
        joints.append(
            {
                "name": joint.name,
                "type": joint_type,
                "parent": model.body(parent_id).name,
                "child": body_name,
                "limits": [float(value) for value in model.jnt_range[index]]
                if model.jnt_limited[index]
                else None,
                "source": "compiled Menagerie MJCF",
            }
        )
    bodies = [model.body(index).name for index in range(model.nbody)]
    actuators = [model.actuator(index).name for index in range(model.nu)]
    return {
        "schema_version": "1.0",
        "robot_id": MODEL_NAME,
        "model_version": f"menagerie-{MODEL_RELEASE}-{record.oid[:12]}",
        "source_artifacts": [
            {
                "repository": "https://github.com/google-deepmind/mujoco_menagerie",
                "package_version": MODEL_RELEASE,
                "object_id": record.oid,
                "entry": record.default_model,
                "license": record.license,
            }
        ],
        "links": bodies,
        "joints": joints,
        "end_effector_candidates": [
            {
                "body": "hand",
                "confidence": 0.95,
                "reason": "Terminal arm body and parent of the two finger branches.",
            }
        ],
        "grippers": [
            {
                "type": "parallel_jaw",
                "finger_joints": ["finger_joint1", "finger_joint2"],
                "actuator": actuators[-1],
                "confidence": 0.95,
            }
        ],
        "confirmed_facts": [
            {"arm_joint_count": 7, "finger_joint_count": 2, "actuator_count": model.nu}
        ],
        "inferences": [
            {"claim": "hand is the reach end-effector", "confidence": 0.95}
        ],
        "capabilities": [
            {"name": "reach", "status": "under_validation"},
            {"name": "parallel_grasp", "status": "unverified"},
        ],
        "discovered_behaviors": [],
        "safety_constraints": [
            {"type": "joint_limits", "source": "compiled Menagerie MJCF"},
            {"type": "simulation_only", "value": True}
        ],
        "unresolved_questions": [
            "Which hand-frame orientation constraints are required for grasping?",
            "Which collision pairs should be treated as allowed grasp contacts?"
        ],
    }
