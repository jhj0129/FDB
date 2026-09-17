from __future__ import annotations

import math
from dataclasses import dataclass

from .inspector import load_robot
from .pose_reach import PandaPoseReachExperiment


@dataclass(frozen=True)
class PlaceCandidate:
    plan_id: str
    place_hand_z: float


@dataclass(frozen=True)
class PickPlaceResult:
    plan: PlaceCandidate
    pick_lift_height_m: float
    final_xy_error_m: float
    final_object_z: float
    released: bool
    stable: bool
    success: bool
    score: float


class PandaPickPlaceExperiment:
    source = (0.45, 0.15, 0.335)
    target = (0.58, -0.12, 0.335)
    grasp_hand_z = 0.375

    def candidates(self) -> tuple[PlaceCandidate, ...]:
        return (
            PlaceCandidate("place_low", 0.365),
            PlaceCandidate("place_centered", 0.375),
            PlaceCandidate("place_high", 0.385),
        )

    def run_candidate(self, plan: PlaceCandidate) -> PickPlaceResult:
        import mujoco

        record, _ = load_robot()
        spec = record.spec(record.default_model)
        spec.worldbody.add_geom(
            name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.5, 0.0, 0.29],
            size=[0.4, 0.4, 0.02], friction=[1.0, 0.005, 0.0001], rgba=[0.5, 0.4, 0.3, 1.0]
        )
        obj = spec.worldbody.add_body(name="object", pos=self.source)
        obj.add_freejoint(name="object_free")
        obj.add_geom(
            name="object_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.022, 0.022, 0.025], density=300,
            friction=[2.0, 0.01, 0.001], rgba=[1.0, 0.1, 0.1, 1.0]
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[:9] = model.key(0).qpos[:9]
        data.ctrl[:] = model.key(0).ctrl
        mujoco.mj_forward(model, data)
        sx, sy, sz = self.source
        tx, ty, tz = self.target
        poses = (
            (sx, sy, 0.52),
            (sx, sy, self.grasp_hand_z),
            (sx, sy, self.grasp_hand_z + 0.20),
            (tx, ty, self.grasp_hand_z + 0.20),
            (tx, ty, plan.place_hand_z),
            (tx, ty, 0.52),
        )
        joint_targets = [
            PandaPoseReachExperiment(pose).run()[0].target_qpos for pose in poses
        ]
        stages = (
            (joint_targets[0], 255.0, 1200),
            (joint_targets[1], 255.0, 1200),
            (joint_targets[1], 0.0, 1200),
            (joint_targets[2], 0.0, 1500),
            (joint_targets[3], 0.0, 1600),
            (joint_targets[4], 0.0, 1400),
            (joint_targets[4], 255.0, 1000),
            (joint_targets[5], 255.0, 1000),
        )
        stable = True
        peak_z = sz
        for target_qpos, gripper, steps in stages:
            data.ctrl[:7] = target_qpos
            data.ctrl[7] = gripper
            for _ in range(steps):
                mujoco.mj_step(model, data)
                peak_z = max(peak_z, float(data.body("object").xpos[2]))
                if not all(math.isfinite(float(value)) for value in data.qpos):
                    stable = False
                    break
        final = data.body("object").xpos.copy()
        xy_error = math.hypot(float(final[0]) - tx, float(final[1]) - ty)
        released = float(data.joint("finger_joint1").qpos[0]) >= 0.035
        success = (
            stable and peak_z - sz >= 0.08 and xy_error <= 0.05
            and abs(float(final[2]) - tz) <= 0.02 and released
        )
        score = (100.0 if success else 0.0) - 200.0 * xy_error - 100.0 * abs(float(final[2]) - tz)
        return PickPlaceResult(
            plan=plan,
            pick_lift_height_m=peak_z - sz,
            final_xy_error_m=xy_error,
            final_object_z=float(final[2]),
            released=released,
            stable=stable,
            success=success,
            score=score,
        )

    def run(self) -> tuple[PickPlaceResult, tuple[PickPlaceResult, ...]]:
        results = tuple(self.run_candidate(candidate) for candidate in self.candidates())
        return max(results, key=lambda item: (item.success, item.score)), results

