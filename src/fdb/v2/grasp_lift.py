from __future__ import annotations

import math
from dataclasses import dataclass

from .inspector import load_robot
from .pose_reach import PandaPoseReachExperiment


@dataclass(frozen=True)
class GraspCandidate:
    plan_id: str
    grasp_hand_z: float


@dataclass(frozen=True)
class GraspLiftResult:
    plan: GraspCandidate
    initial_object_z: float
    final_object_z: float
    lift_height_m: float
    final_object_to_hand_distance_m: float
    forbidden_contact_steps: int
    stable: bool
    success: bool
    score: float


class PandaGraspLiftExperiment:
    object_position = (0.45, 0.15, 0.335)

    def candidates(self) -> tuple[GraspCandidate, ...]:
        return (
            GraspCandidate("grasp_low", 0.375),
            GraspCandidate("grasp_centered", 0.385),
            GraspCandidate("grasp_high", 0.395),
        )

    def run_candidate(self, plan: GraspCandidate) -> GraspLiftResult:
        import mujoco

        record, _ = load_robot()
        spec = record.spec(record.default_model)
        spec.worldbody.add_geom(
            name="table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.45, 0.15, 0.29],
            size=[0.35, 0.35, 0.02],
            friction=[1.0, 0.005, 0.0001],
            rgba=[0.5, 0.4, 0.3, 1.0],
        )
        obj = spec.worldbody.add_body(name="object", pos=self.object_position)
        obj.add_freejoint(name="object_free")
        obj.add_geom(
            name="object_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.022, 0.022, 0.025],
            density=300,
            friction=[2.0, 0.01, 0.001],
            rgba=[1.0, 0.1, 0.1, 1.0],
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        # The original keyframe has no coordinates for the added free joint. Preserve
        # model qpos0 for the object and copy only the original robot home coordinates.
        mujoco.mj_resetData(model, data)
        data.qpos[:9] = model.key(0).qpos[:9]
        data.ctrl[:] = model.key(0).ctrl
        mujoco.mj_forward(model, data)

        x, y, initial_z = self.object_position
        pregrasp = (x, y, 0.52)
        grasp = (x, y, plan.grasp_hand_z)
        lift = (x, y, plan.grasp_hand_z + 0.20)
        joint_targets = [
            PandaPoseReachExperiment(target).run()[0].target_qpos
            for target in (pregrasp, grasp, lift)
        ]
        stages = (
            (joint_targets[0], 255.0, 1200, False),
            (joint_targets[1], 255.0, 1200, False),
            (joint_targets[1], 0.0, 1200, False),
            (joint_targets[2], 0.0, 1500, True),
        )
        forbidden_contacts = 0
        stable = True
        lifted_once = False
        table_id = model.geom("table").id
        object_id = model.geom("object_geom").id
        for joint_target, gripper_control, steps, lifting in stages:
            data.ctrl[:7] = joint_target
            data.ctrl[7] = gripper_control
            for _ in range(steps):
                mujoco.mj_step(model, data)
                if not all(math.isfinite(float(value)) for value in data.qpos):
                    stable = False
                    break
                if lifting:
                    if float(data.body("object").xpos[2]) >= initial_z + 0.02:
                        lifted_once = True
                    for index in range(data.ncon):
                        pair = {int(data.contact[index].geom1), int(data.contact[index].geom2)}
                        if lifted_once and pair == {table_id, object_id}:
                            forbidden_contacts += 1

        final_object = data.body("object").xpos.copy()
        final_hand = data.body("hand").xpos.copy()
        lift_height = float(final_object[2] - initial_z)
        retention_distance = float(
            math.sqrt(sum((float(final_object[i]) - float(final_hand[i])) ** 2 for i in range(3)))
        )
        success = (
            stable
            and lift_height >= 0.08
            and retention_distance <= 0.16
            and forbidden_contacts == 0
        )
        score = (
            (100.0 if success else 0.0)
            + 100.0 * lift_height
            - 20.0 * retention_distance
            - 2.0 * forbidden_contacts
        )
        return GraspLiftResult(
            plan=plan,
            initial_object_z=initial_z,
            final_object_z=float(final_object[2]),
            lift_height_m=lift_height,
            final_object_to_hand_distance_m=retention_distance,
            forbidden_contact_steps=forbidden_contacts,
            stable=stable,
            success=success,
            score=score,
        )

    def run(self) -> tuple[GraspLiftResult, tuple[GraspLiftResult, ...]]:
        results = tuple(self.run_candidate(candidate) for candidate in self.candidates())
        return max(results, key=lambda item: (item.success, item.score)), results
