from __future__ import annotations

import math
from dataclasses import dataclass

from .inspector import load_robot
from .reach import ReachCandidate


@dataclass(frozen=True)
class PoseReachResult:
    plan: ReachCandidate
    converged: bool
    iterations: int
    position_error_m: float
    orientation_error_rad: float
    joint_limit_violations: int
    target_qpos: tuple[float, ...]
    score: float


class PandaPoseReachExperiment:
    """Six-dimensional hand-pose IK using MuJoCo position and rotation Jacobians."""

    def __init__(
        self,
        target_position: tuple[float, float, float] = (0.45, 0.15, 0.55),
        target_quaternion: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.target_position = target_position
        self.target_quaternion = target_quaternion

    def run_candidate(self, plan: ReachCandidate) -> PoseReachResult:
        import mujoco
        import numpy as np

        _, model = load_robot()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        hand_id = model.body("hand").id
        desired_position = np.asarray(self.target_position, dtype=float)
        desired_quaternion = (
            np.asarray(self.target_quaternion, dtype=float)
            if self.target_quaternion is not None
            else data.body("hand").xquat.copy()
        )
        desired_quaternion /= np.linalg.norm(desired_quaternion)
        desired_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(desired_rotation, desired_quaternion)
        desired_rotation = desired_rotation.reshape(3, 3)
        arm_joint_ids = [model.joint(f"joint{index}").id for index in range(1, 8)]
        converged = False
        position_error = math.inf
        orientation_error = math.inf
        iterations = 0
        for iterations in range(1, plan.max_iterations + 1):
            translation = desired_position - data.body("hand").xpos.copy()
            current_rotation = data.body("hand").xmat.reshape(3, 3)
            rotation = 0.5 * sum(
                np.cross(current_rotation[:, axis], desired_rotation[:, axis])
                for axis in range(3)
            )
            position_error = float(np.linalg.norm(translation))
            orientation_error = float(np.linalg.norm(rotation))
            if position_error <= 1e-4 and orientation_error <= 1e-3:
                converged = True
                break
            jac_position = np.zeros((3, model.nv))
            jac_rotation = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jac_position, jac_rotation, hand_id)
            jacobian = np.vstack((jac_position[:, :7], jac_rotation[:, :7]))
            error = np.concatenate((translation, rotation))
            system = jacobian @ jacobian.T + (plan.damping**2) * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(system, error)
            norm = float(np.linalg.norm(delta))
            if norm > 0.12:
                delta *= 0.12 / norm
            data.qpos[:7] += delta
            for joint_id in arm_joint_ids:
                address = model.jnt_qposadr[joint_id]
                low, high = model.jnt_range[joint_id]
                data.qpos[address] = np.clip(data.qpos[address], low, high)
            mujoco.mj_forward(model, data)

        target_qpos = data.qpos[:7].copy()
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.ctrl[:7] = target_qpos
        data.ctrl[7] = 255.0
        for _ in range(1500):
            mujoco.mj_step(model, data)
        position_error = float(
            np.linalg.norm(desired_position - data.body("hand").xpos.copy())
        )
        rotation = np.zeros(3)
        mujoco.mju_subQuat(rotation, desired_quaternion, data.body("hand").xquat)
        orientation_error = float(np.linalg.norm(rotation))
        violations = sum(
            not (
                model.jnt_range[joint_id][0] - 1e-8
                <= data.qpos[model.jnt_qposadr[joint_id]]
                <= model.jnt_range[joint_id][1] + 1e-8
            )
            for joint_id in arm_joint_ids
        )
        success = (
            converged
            and position_error <= 0.02
            and orientation_error <= 0.03
            and violations == 0
        )
        score = (
            (100.0 if success else 0.0)
            - 500.0 * position_error
            - 50.0 * orientation_error
            - 25.0 * violations
        )
        return PoseReachResult(
            plan=plan,
            converged=converged,
            iterations=iterations,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            joint_limit_violations=violations,
            target_qpos=tuple(float(value) for value in target_qpos),
            score=score,
        )

    def run(self) -> tuple[PoseReachResult, tuple[PoseReachResult, ...]]:
        candidates = (
            ReachCandidate("pose_low_damping", 0.01, 300),
            ReachCandidate("pose_balanced_damping", 0.05, 300),
            ReachCandidate("pose_high_damping", 0.15, 300),
        )
        results = tuple(self.run_candidate(candidate) for candidate in candidates)
        return max(results, key=lambda item: (item.score, -item.iterations)), results
