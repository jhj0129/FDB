from __future__ import annotations

from typing import Any


class GenericMorphologyInspector:
    """Derive a conservative self-model without robot-specific link names."""

    def inspect_menagerie(self, model_name: str) -> dict[str, Any]:
        import mujoco
        import mujoco_menagerie as menagerie

        source = menagerie.get(model_name)
        model = source.model(source.default_model)
        child_count = [0] * model.nbody
        for body_id in range(1, model.nbody):
            child_count[int(model.body_parentid[body_id])] += 1

        links = []
        for body_id in range(1, model.nbody):
            parent_id = int(model.body_parentid[body_id])
            links.append(
                {
                    "name": model.body(body_id).name,
                    "parent": model.body(parent_id).name,
                    "child_count": child_count[body_id],
                    "mass": float(model.body_mass[body_id]),
                }
            )

        joints = []
        for joint_id in range(model.njnt):
            body_id = int(model.jnt_bodyid[joint_id])
            parent_id = int(model.body_parentid[body_id])
            joints.append(
                {
                    "name": model.joint(joint_id).name,
                    "type": mujoco.mjtJoint(int(model.jnt_type[joint_id])).name.removeprefix("mjJNT_").lower(),
                    "parent": model.body(parent_id).name,
                    "child": model.body(body_id).name,
                    "limits": [float(v) for v in model.jnt_range[joint_id]]
                    if model.jnt_limited[joint_id]
                    else None,
                    "axis": [float(v) for v in model.jnt_axis[joint_id]],
                    "source": "compiled Menagerie MJCF",
                }
            )

        end_effectors = []
        for site_id in range(model.nsite):
            site_name = model.site(site_id).name or f"site_{site_id}"
            body_name = model.body(int(model.site_bodyid[site_id])).name
            confidence = 0.95 if any(token in site_name.lower() for token in ("attach", "grip", "tcp", "ee")) else 0.65
            end_effectors.append(
                {"site": site_name, "body": body_name, "confidence": confidence, "reason": "Terminal/tool site from compiled model."}
            )
        if not end_effectors:
            for body_id in range(1, model.nbody):
                if child_count[body_id] == 0:
                    end_effectors.append(
                        {"body": model.body(body_id).name, "confidence": 0.55, "reason": "Leaf body; requires behavioral verification."}
                    )

        actuators = []
        actuated_joint_ids: set[int] = set()
        for actuator_id in range(model.nu):
            transmission_type = mujoco.mjtTrn(int(model.actuator_trntype[actuator_id])).name.removeprefix("mjTRN_").lower()
            target_id = int(model.actuator_trnid[actuator_id, 0])
            if transmission_type in ("joint", "jointinparent"):
                actuated_joint_ids.add(target_id)
            actuators.append(
                {
                    "name": model.actuator(actuator_id).name,
                    "transmission": transmission_type,
                    "target_id": target_id,
                    "control_range": [float(v) for v in model.actuator_ctrlrange[actuator_id]],
                }
            )

        finger_joints = [
            joint["name"] for joint in joints
            if "finger" in (joint["name"] or "").lower() or "gripper" in (joint["name"] or "").lower()
        ]
        grippers = []
        if finger_joints:
            grippers.append(
                {"type": "candidate_parallel_or_multi_finger", "finger_joints": finger_joints, "confidence": 0.8}
            )
            finger_body_ids = {
                int(model.jnt_bodyid[model.joint(name).id]) for name in finger_joints
            }
            common_parents = {
                int(model.body_parentid[body_id]) for body_id in finger_body_ids
            }
            if len(common_parents) == 1:
                parent_id = common_parents.pop()
                end_effectors.insert(
                    0,
                    {
                        "body": model.body(parent_id).name,
                        "confidence": 0.9,
                        "reason": "Common parent of detected finger-joint bodies.",
                    },
                )

        arm_joint_count = sum(
            1 for joint_id in actuated_joint_ids
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        )
        return {
            "schema_version": "1.0",
            "robot_id": model_name,
            "model_version": f"menagerie-{menagerie.__version__}-{source.oid[:12]}",
            "source_artifacts": [{"repository": "https://github.com/google-deepmind/mujoco_menagerie", "object_id": source.oid, "entry": source.default_model, "license": source.license}],
            "links": links,
            "joints": joints,
            "end_effector_candidates": end_effectors,
            "grippers": grippers,
            "confirmed_facts": [{"joint_count": model.njnt, "actuator_count": model.nu, "actuated_hinge_count": arm_joint_count, "actuators": actuators}],
            "inferences": [{"claim": "serial_arm_candidate", "confidence": 0.9 if arm_joint_count >= 6 else 0.4}],
            "capabilities": [{"name": "position_reach_candidate", "status": "unverified"}] if arm_joint_count >= 6 else [],
            "discovered_behaviors": [],
            "safety_constraints": [{"type": "joint_limits", "source": "compiled model"}, {"type": "simulation_only", "value": True}],
            "unresolved_questions": ["Verify end-effector behavior with small simulation motions.", "Determine collision-safe home configuration."],
        }
