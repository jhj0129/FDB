from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class JointProbeResult:
    joint: str
    delta_rad: float
    predicted_downstream_bodies: tuple[str, ...]
    observed_affected_bodies: tuple[str, ...]
    unexpected_affected_bodies: tuple[str, ...]
    verified: bool


@dataclass(frozen=True)
class RobotProbeResult:
    robot: str
    joint_results: tuple[JointProbeResult, ...]
    verified_count: int
    probe_count: int
    success_rate: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MotionProbe:
    def probe_menagerie(self, model_name: str, delta_rad: float = 0.02) -> RobotProbeResult:
        import mujoco
        import mujoco_menagerie as menagerie
        import numpy as np

        model = menagerie.get(model_name).model()
        children: dict[int, list[int]] = {body_id: [] for body_id in range(model.nbody)}
        for body_id in range(1, model.nbody):
            children[int(model.body_parentid[body_id])].append(body_id)

        def descendants(root: int) -> set[int]:
            found = {root}
            pending = [root]
            while pending:
                child_ids = children[pending.pop()]
                found.update(child_ids)
                pending.extend(child_ids)
            return found

        actuated_joint_ids = {
            int(model.actuator_trnid[index, 0])
            for index in range(model.nu)
            if int(model.actuator_trntype[index])
            in (
                int(mujoco.mjtTrn.mjTRN_JOINT),
                int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
            )
        }
        results: list[JointProbeResult] = []
        for joint_id in sorted(actuated_joint_ids):
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                continue
            data = mujoco.MjData(model)
            if model.nkey:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            else:
                mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            before_position = data.xpos.copy()
            before_quaternion = data.xquat.copy()
            address = int(model.jnt_qposadr[joint_id])
            applied_delta = delta_rad
            if model.jnt_limited[joint_id]:
                low, high = model.jnt_range[joint_id]
                if data.qpos[address] + applied_delta > high:
                    applied_delta = -delta_rad
                data.qpos[address] = np.clip(
                    data.qpos[address] + applied_delta, low, high
                )
            else:
                data.qpos[address] += applied_delta
            mujoco.mj_forward(model, data)
            affected: set[int] = set()
            for body_id in range(1, model.nbody):
                position_delta = float(
                    np.linalg.norm(data.xpos[body_id] - before_position[body_id])
                )
                rotation_delta = np.zeros(3)
                mujoco.mju_subQuat(
                    rotation_delta,
                    data.xquat[body_id],
                    before_quaternion[body_id],
                )
                if position_delta > 1e-8 or float(np.linalg.norm(rotation_delta)) > 1e-8:
                    affected.add(body_id)
            joint_body = int(model.jnt_bodyid[joint_id])
            predicted = descendants(joint_body)
            unexpected = affected - predicted
            verified = not unexpected and bool(affected & predicted)
            results.append(
                JointProbeResult(
                    joint=model.joint(joint_id).name,
                    delta_rad=applied_delta,
                    predicted_downstream_bodies=tuple(
                        model.body(body_id).name for body_id in sorted(predicted)
                    ),
                    observed_affected_bodies=tuple(
                        model.body(body_id).name for body_id in sorted(affected)
                    ),
                    unexpected_affected_bodies=tuple(
                        model.body(body_id).name for body_id in sorted(unexpected)
                    ),
                    verified=verified,
                )
            )
        verified_count = sum(result.verified for result in results)
        return RobotProbeResult(
            robot=model_name,
            joint_results=tuple(results),
            verified_count=verified_count,
            probe_count=len(results),
            success_rate=verified_count / len(results),
        )
