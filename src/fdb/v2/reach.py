from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .inspector import load_robot


@dataclass(frozen=True)
class ReachCandidate:
    plan_id: str
    damping: float
    max_iterations: int = 200


@dataclass(frozen=True)
class ReachResult:
    plan: ReachCandidate
    converged: bool
    ik_iterations: int
    ik_error_m: float
    execution_error_m: float
    joint_limit_violations: int
    contact_count: int
    score: float
    target_qpos: tuple[float, ...]


class PandaReachExperiment:
    def __init__(self, target: tuple[float, float, float] = (0.45, 0.15, 0.55)) -> None:
        self.target = target

    def candidates(self) -> tuple[ReachCandidate, ...]:
        return (
            ReachCandidate("low_damping", 0.005),
            ReachCandidate("balanced_damping", 0.03),
            ReachCandidate("high_damping", 0.15),
        )

    def run_candidate(self, plan: ReachCandidate) -> ReachResult:
        import mujoco
        import numpy as np

        _, model = load_robot()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        hand_id = model.body("hand").id
        arm_joint_ids = [model.joint(f"joint{index}").id for index in range(1, 8)]
        target = np.asarray(self.target, dtype=float)
        converged = False
        iterations = 0
        error_norm = math.inf
        for iterations in range(1, plan.max_iterations + 1):
            error = target - data.body("hand").xpos.copy()
            error_norm = float(np.linalg.norm(error))
            if error_norm <= 1e-4:
                converged = True
                break
            jac_position = np.zeros((3, model.nv))
            jac_rotation = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jac_position, jac_rotation, hand_id)
            jacobian = jac_position[:, :7]
            system = jacobian @ jacobian.T + (plan.damping**2) * np.eye(3)
            delta = jacobian.T @ np.linalg.solve(system, error)
            norm = float(np.linalg.norm(delta))
            if norm > 0.12:
                delta *= 0.12 / norm
            data.qpos[:7] += delta
            for joint_id in arm_joint_ids:
                if model.jnt_limited[joint_id]:
                    address = model.jnt_qposadr[joint_id]
                    low, high = model.jnt_range[joint_id]
                    data.qpos[address] = np.clip(data.qpos[address], low, high)
            mujoco.mj_forward(model, data)

        target_qpos = data.qpos[:7].copy()
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.ctrl[:7] = target_qpos
        data.ctrl[7] = 255.0
        contact_count = 0
        for _ in range(1500):
            mujoco.mj_step(model, data)
            contact_count += data.ncon
        execution_error = float(np.linalg.norm(target - data.body("hand").xpos.copy()))
        violations = 0
        for joint_id in arm_joint_ids:
            address = model.jnt_qposadr[joint_id]
            low, high = model.jnt_range[joint_id]
            if not (low - 1e-8 <= data.qpos[address] <= high + 1e-8):
                violations += 1
        success = converged and execution_error <= 0.02 and violations == 0
        score = (100.0 if success else 0.0) - 500.0 * execution_error - violations * 25
        return ReachResult(
            plan=plan,
            converged=converged,
            ik_iterations=iterations,
            ik_error_m=error_norm,
            execution_error_m=execution_error,
            joint_limit_violations=violations,
            contact_count=contact_count,
            score=score,
            target_qpos=tuple(float(value) for value in target_qpos),
        )

    def run(self) -> tuple[ReachResult, tuple[ReachResult, ...]]:
        results = tuple(self.run_candidate(candidate) for candidate in self.candidates())
        selected = max(
            results,
            key=lambda item: (
                item.converged and item.execution_error_m <= 0.02,
                item.score,
                -item.ik_iterations,
            ),
        )
        return selected, results

    @staticmethod
    def as_dict(result: ReachResult) -> dict[str, object]:
        return asdict(result)

