from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .inspector import load_robot
from .pose_reach import PandaPoseReachExperiment


@dataclass(frozen=True)
class PickPlaceScenario:
    scenario_id: str = "nominal"
    source_xy: tuple[float, float] = (0.45, 0.15)
    target_xy: tuple[float, float] = (0.58, -0.12)
    half_size: tuple[float, float, float] = (0.022, 0.022, 0.025)
    density: float = 300.0
    sliding_friction: float = 2.0


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
    robot_table_contact_steps: int
    deepest_robot_table_penetration_m: float
    max_robot_table_contact_force_n: float
    robot_table_contact_pairs: tuple[tuple[str, int], ...]
    minimum_hand_table_clearance_m: float
    stage_sequence_valid: bool
    stage_violations: tuple[str, ...]
    success: bool
    score: float


class PandaPickPlaceExperiment:
    def __init__(
        self,
        scenario: PickPlaceScenario | None = None,
        *,
        grasp_clearance_m: float = 0.10,
    ) -> None:
        self.scenario = scenario or PickPlaceScenario()
        object_z = 0.31 + self.scenario.half_size[2]
        self.source = (*self.scenario.source_xy, object_z)
        self.target = (*self.scenario.target_xy, object_z)
        # The original +0.04 m pose let the Panda finger collision meshes penetrate
        # the table. Keep the fingertips in the object's upper half instead.
        self.grasp_hand_z = object_z + grasp_clearance_m

    def candidates(self) -> tuple[PlaceCandidate, ...]:
        return (
            PlaceCandidate("place_low", 0.445),
            PlaceCandidate("place_centered", 0.455),
            PlaceCandidate("place_high", 0.465),
        )

    def run_candidate(self, plan: PlaceCandidate) -> PickPlaceResult:
        import mujoco
        import numpy as np

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
            size=self.scenario.half_size, density=self.scenario.density,
            friction=[self.scenario.sliding_friction, 0.01, 0.001], rgba=[1.0, 0.1, 0.1, 1.0]
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
        table_id = model.geom("table").id
        object_id = model.geom("object_geom").id
        robot_table_contact_steps = 0
        deepest_penetration = 0.0
        max_contact_force = 0.0
        contact_pairs: Counter[str] = Counter()
        minimum_hand_clearance = math.inf
        stage_sequence_valid = True
        stage_violations: list[str] = []
        stage_names = ("pregrasp", "descend", "grasp", "lift", "transfer", "lower", "release", "retreat")
        for stage_name, (target_qpos, gripper, steps) in zip(stage_names, stages):
            if stage_name == "transfer" and float(data.body("object").xpos[2]) - sz < 0.08:
                stage_sequence_valid = False
                stage_violations.append("안전 높이에 도달하기 전에 횡이동 시작")
            if stage_name == "release" and abs(float(data.body("object").xpos[2]) - tz) > 0.04:
                stage_sequence_valid = False
                stage_violations.append("물체가 테이블 지지 높이에 도달하기 전에 그리퍼 개방")
            if stage_name == "retreat" and float(data.joint("finger_joint1").qpos[0]) < 0.03:
                stage_sequence_valid = False
                stage_violations.append("그리퍼 개방 완료 전에 후퇴")
            start_arm = data.ctrl[:7].copy()
            start_gripper = float(data.ctrl[7])
            for step in range(steps):
                phase = (step + 1) / steps
                blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
                data.ctrl[:7] = start_arm + blend * (target_qpos - start_arm)
                data.ctrl[7] = start_gripper + blend * (gripper - start_gripper)
                mujoco.mj_step(model, data)
                peak_z = max(peak_z, float(data.body("object").xpos[2]))
                minimum_hand_clearance = min(
                    minimum_hand_clearance,
                    float(data.body("hand").xpos[2]) - 0.31,
                )
                contacted_this_step = False
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if table_id in pair and object_id not in pair:
                        contacted_this_step = True
                        other_id = int(contact.geom2) if int(contact.geom1) == table_id else int(contact.geom1)
                        body_name = model.body(int(model.geom_bodyid[other_id])).name
                        contact_name = model.geom(other_id).name or f"{body_name}/geom_{other_id}"
                        contact_pairs[contact_name] += 1
                        deepest_penetration = max(deepest_penetration, max(0.0, -float(contact.dist)))
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, contact_index, force)
                        max_contact_force = max(max_contact_force, abs(float(force[0])))
                if contacted_this_step:
                    robot_table_contact_steps += 1
                if not all(math.isfinite(float(value)) for value in data.qpos):
                    stable = False
                    break
        final = data.body("object").xpos.copy()
        xy_error = math.hypot(float(final[0]) - tx, float(final[1]) - ty)
        released = float(data.joint("finger_joint1").qpos[0]) >= 0.035
        success = (
            stable and peak_z - sz >= 0.08 and xy_error <= 0.05
            and abs(float(final[2]) - tz) <= 0.02 and released
            and robot_table_contact_steps == 0
            and minimum_hand_clearance >= 0.055
            and stage_sequence_valid
        )
        score = (
            (100.0 if success else 0.0)
            - 200.0 * xy_error
            - 100.0 * abs(float(final[2]) - tz)
            - 500.0 * robot_table_contact_steps
        )
        return PickPlaceResult(
            plan=plan,
            pick_lift_height_m=peak_z - sz,
            final_xy_error_m=xy_error,
            final_object_z=float(final[2]),
            released=released,
            stable=stable,
            robot_table_contact_steps=robot_table_contact_steps,
            deepest_robot_table_penetration_m=deepest_penetration,
            max_robot_table_contact_force_n=max_contact_force,
            robot_table_contact_pairs=tuple(contact_pairs.most_common()),
            minimum_hand_table_clearance_m=minimum_hand_clearance,
            stage_sequence_valid=stage_sequence_valid,
            stage_violations=tuple(stage_violations),
            success=success,
            score=score,
        )

    def run(self) -> tuple[PickPlaceResult, tuple[PickPlaceResult, ...]]:
        results = tuple(self.run_candidate(candidate) for candidate in self.candidates())
        return max(
            results,
            key=lambda item: (
                item.robot_table_contact_steps == 0,
                item.stage_sequence_valid,
                item.success,
                item.score,
            ),
        ), results
