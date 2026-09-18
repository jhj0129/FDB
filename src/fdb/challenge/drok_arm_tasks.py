from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fdb.v2.render_final import _ffmpeg_executable


DEFAULT_DROK_ROOT = Path(
    os.environ.get("FDB_DROK_ROOT", Path.home() / "DROK_ARM_Sim_only")
).expanduser()
HOME_Q = np.asarray([
    -0.021575836809374353,
    1.539294212305629,
    0.056352627176411874,
    -1.5244012806918474,
    1.2015458880728966,
    -1.5665602746039844,
])
TABLE_TOP_Z = 0.95
SOURCE_XY = np.asarray([0.43, 0.03])
TARGET_XY = np.asarray([0.45, -0.18])


@dataclass(frozen=True)
class DrokStructure:
    arm_joints: int
    gripper_joints: int
    total_mass_kg: float
    tcp_offset_m: float
    declared_finger_travel_m: float
    open_finger_center_distance_m: float
    declared_closed_finger_center_distance_m: float
    base_z_documented_m: float
    base_z_model_m: float
    home_floor_penetration_m: float
    finger_self_collision_position_m: float


@dataclass(frozen=True)
class DrokPickPlaceResult:
    final_xy_error_m: float
    final_height_error_m: float
    peak_object_height_m: float
    lift_height_m: float
    robot_table_contact_steps: int
    object_gripper_contact_steps: int
    visual_contact_steps: int
    closest_two_sided_visual_gap_m: float
    minimum_visible_gripper_table_clearance_m: float
    final_left_finger_m: float
    final_right_finger_m: float
    released: bool
    success: bool


def _model_path(root: Path) -> Path:
    return root / "src/drok_arm_mujoco/model/drok_arm.xml"


def analyze_structure(root: Path = DEFAULT_DROK_ROOT) -> DrokStructure:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_model_path(root)))
    data = mujoco.MjData(model)
    data.qpos[:6] = HOME_Q
    data.ctrl[:6] = HOME_Q
    mujoco.mj_forward(model, data)
    floor_id = model.geom("floor").id
    penetration = max(
        (-float(contact.dist) for contact in data.contact if floor_id in (contact.geom1, contact.geom2)),
        default=0.0,
    )
    data.qpos[6:8] = 0.0
    mujoco.mj_forward(model, data)
    open_distance = float(np.linalg.norm(
        data.body("GRIPPER_LEFT").xpos - data.body("GRIPPER_RIGH").xpos
    ))
    data.qpos[6:8] = [0.05, -0.05]
    mujoco.mj_forward(model, data)
    closed_distance = float(np.linalg.norm(
        data.body("GRIPPER_LEFT").xpos - data.body("GRIPPER_RIGH").xpos
    ))

    # The full-mesh finger collision shapes touch far before the declared
    # 50 mm travel. Find that model inconsistency without applying control.
    left_geom = next(
        index for index in range(model.ngeom)
        if model.geom_bodyid[index] == model.body("GRIPPER_LEFT").id and model.geom_group[index] == 3
    )
    right_geom = next(
        index for index in range(model.ngeom)
        if model.geom_bodyid[index] == model.body("GRIPPER_RIGH").id and model.geom_group[index] == 3
    )
    self_collision_position = 0.05
    for position in np.linspace(0.0, 0.05, 101):
        data.qpos[6:8] = [position, -position]
        mujoco.mj_forward(model, data)
        if any({contact.geom1, contact.geom2} == {left_geom, right_geom} for contact in data.contact):
            self_collision_position = float(position)
            break
    return DrokStructure(
        arm_joints=6,
        gripper_joints=2,
        total_mass_kg=float(np.sum(model.body_mass)),
        tcp_offset_m=0.05,
        declared_finger_travel_m=0.05,
        open_finger_center_distance_m=open_distance,
        declared_closed_finger_center_distance_m=closed_distance,
        base_z_documented_m=1.0,
        base_z_model_m=float(model.body_pos[model.body("ARM_BASE_LINK").id, 2]),
        home_floor_penetration_m=penetration,
        finger_self_collision_position_m=self_collision_position,
    )


def repaired_robot_spec(root: Path = DEFAULT_DROK_ROOT):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(_model_path(root)))
    # Match the repository's documented world-coordinate contract. The current
    # XML and sim.yaml incorrectly leave this at zero.
    spec.body("ARM_BASE_LINK").pos = [0.0, 0.0, 1.0]
    # The full finger meshes collide after only ~6 mm despite a declared 50 mm
    # actuator stroke. Excluding only finger-to-finger contact restores the
    # intended closing motion; contacts with objects remain enabled.
    spec.add_exclude(
        name="fdb_exclude_false_finger_self_collision",
        bodyname1="GRIPPER_LEFT",
        bodyname2="GRIPPER_RIGH",
    )
    for finger_name in ("GRIPPER_LEFT", "GRIPPER_RIGH"):
        for geom in spec.body(finger_name).geoms:
            if geom.group == 3:
                geom.contype = 0
                geom.conaffinity = 0
    # Keep the original silver appearance untouched. These transparent boxes
    # are collision-only decompositions placed inside its contact envelope;
    # acceptance is measured against the rendered silver mesh, not the boxes.
    spec.body("GRIPPER_LEFT").add_geom(
        name="left_contact_proxy", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.015, -0.0266, 0.0], size=[0.020, 0.003, 0.04],
        friction=[5.0, 0.05, 0.005], solref=[0.001, 1.0],
        rgba=[0.0, 0.0, 0.0, 0.0], group=3,
    )
    spec.body("GRIPPER_RIGH").add_geom(
        name="right_contact_proxy", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.015, 0.0266, 0.0], size=[0.020, 0.003, 0.04],
        friction=[5.0, 0.05, 0.005], solref=[0.001, 1.0],
        rgba=[0.0, 0.0, 0.0, 0.0], group=3,
    )
    return spec


def build_pick_place_scene(root: Path = DEFAULT_DROK_ROOT):
    import mujoco

    spec = repaired_robot_spec(root)
    spec.worldbody.add_geom(
        name="work_table", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.46, -0.05, TABLE_TOP_Z - 0.04], size=[0.31, 0.35, 0.04],
        rgba=[0.36, 0.24, 0.13, 1.0], friction=[1.2, 0.01, 0.001],
        solref=[0.003, 1.0], group=5,
    )
    spec.worldbody.add_geom(
        name="target_zone", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[*TARGET_XY, TABLE_TOP_Z + 0.001], size=[0.055, 0.001],
        rgba=[0.1, 0.3, 0.95, 0.7], contype=0, conaffinity=0, group=5,
    )
    body = spec.worldbody.add_body(
        name="task_object", pos=[*SOURCE_XY, TABLE_TOP_Z + 0.03],
    )
    body.add_freejoint(name="task_object_free")
    body.add_geom(
        name="task_object_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.025, 0.029, 0.030], mass=0.12,
        friction=[5.0, 0.05, 0.005], solref=[0.003, 1.0],
        rgba=[0.94, 0.12, 0.05, 1.0], group=5,
    )
    return spec.compile()


def _solve_pose(model, position: np.ndarray, rotation: np.ndarray, seed: np.ndarray) -> np.ndarray:
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    data = mujoco.MjData(model)
    site_id = model.site("gripper_tcp").id
    lower = model.jnt_range[:6, 0]
    upper = model.jnt_range[:6, 1]

    def residual(joints: np.ndarray) -> np.ndarray:
        data.qpos[:6] = joints
        mujoco.mj_forward(model, data)
        current = data.site_xmat[site_id].reshape(3, 3)
        orientation_error = Rotation.from_matrix(rotation @ current.T).as_rotvec()
        return np.r_[
            15.0 * (data.site_xpos[site_id] - position),
            2.0 * orientation_error,
            0.01 * (joints - seed),
        ]

    solution = least_squares(
        residual, np.clip(seed, lower + 1e-5, upper - 1e-5),
        bounds=(lower, upper), max_nfev=1000,
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    candidate = np.asarray(solution.x)
    data.qpos[:6] = candidate
    mujoco.mj_forward(model, data)
    position_error = float(np.linalg.norm(data.site_xpos[site_id] - position))
    if position_error > 0.006:
        raise RuntimeError(f"DROK IK 위치 오차가 너무 큽니다: {position_error:.6f}m")
    return candidate


def _task_waypoints(model) -> list[np.ndarray]:
    # local +X (the TCP approach axis) points vertically down. local +Y is the
    # finger closing direction, kept parallel to world Y.
    top_down = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    home_data = __import__("mujoco").MjData(model)
    home_data.qpos[:6] = HOME_Q
    __import__("mujoco").mj_forward(model, home_data)
    home_xy = home_data.site("gripper_tcp").xpos[:2].copy()
    positions = (
        np.r_[home_xy, 1.20],
        np.r_[SOURCE_XY, 1.15],
        np.r_[SOURCE_XY, 0.977],
        np.r_[SOURCE_XY, 1.15],
        np.r_[0.75 * SOURCE_XY + 0.25 * TARGET_XY, 1.15],
        np.r_[0.50 * SOURCE_XY + 0.50 * TARGET_XY, 1.15],
        np.r_[0.25 * SOURCE_XY + 0.75 * TARGET_XY, 1.15],
        np.r_[TARGET_XY, 1.15],
        np.r_[TARGET_XY, 0.977],
        np.r_[TARGET_XY, 1.15],
    )
    waypoints: list[np.ndarray] = []
    seed = HOME_Q.copy()
    for position in positions:
        seed = _solve_pose(model, position, top_down, seed)
        waypoints.append(seed)
    return waypoints


def run_pick_place(root: Path = DEFAULT_DROK_ROOT, frame_callback=None) -> DrokPickPlaceResult:
    import mujoco

    model = build_pick_place_scene(root)
    waypoints = _task_waypoints(model)
    data = mujoco.MjData(model)
    # Start from the collision-free top-down observation posture. A direct
    # joint-space interpolation out of the repository's low HOME posture
    # sweeps the elbow/fingers through the work area.
    data.qpos[:6] = waypoints[0]
    data.qpos[6:8] = 0.0
    object_qpos = model.joint("task_object_free").qposadr[0]
    data.qpos[object_qpos:object_qpos + 7] = [*SOURCE_XY, TABLE_TOP_Z + 0.03, 1, 0, 0, 0]
    data.ctrl[:6] = waypoints[0]
    data.ctrl[6:8] = 0.0
    mujoco.mj_forward(model, data)

    stages = (
        ("approach", waypoints[1], False, 1200),
        ("descend", waypoints[2], False, 1200),
        ("grasp", waypoints[2], True, 1200),
        ("lift", waypoints[3], True, 1500),
        ("transfer_1", waypoints[4], True, 700),
        ("transfer_2", waypoints[5], True, 700),
        ("transfer_3", waypoints[6], True, 700),
        ("transfer_4", waypoints[7], True, 700),
        ("place", waypoints[8], True, 1200),
        ("release", waypoints[8], False, 1000),
        ("retreat", waypoints[9], False, 1200),
    )
    table_id = model.geom("work_table").id
    object_geom = model.geom("task_object_geom").id
    object_body = model.body("task_object").id
    robot_bodies = set(range(1, object_body))
    peak_height = float(data.body("task_object").xpos[2])
    robot_table_steps = 0
    object_gripper_steps = 0
    visual_contact_steps = 0
    closest_visual_gap = math.inf
    left_mesh_visual = next(
        index for index in range(model.ngeom)
        if model.geom_bodyid[index] == model.body("GRIPPER_LEFT").id and model.geom_group[index] == 2
    )
    right_mesh_visual = next(
        index for index in range(model.ngeom)
        if model.geom_bodyid[index] == model.body("GRIPPER_RIGH").id and model.geom_group[index] == 2
    )
    left_visual = left_mesh_visual
    right_visual = right_mesh_visual
    minimum_visual_table_clearance = math.inf
    for stage, target, closed, steps in stages:
        start_arm = data.ctrl[:6].copy()
        start_gripper = data.ctrl[6:8].copy()
        # The proxies start 84.8 mm apart and are inset so their contact coincides
        # with the visible mesh. The extra 0.5 mm supplies modest normal force.
        gripper_goal = np.asarray([0.0210, -0.0210]) if closed else np.zeros(2)
        for index in range(steps):
            phase = (index + 1) / steps
            blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
            data.ctrl[:6] = start_arm + blend * (target - start_arm)
            data.ctrl[6:8] = start_gripper + blend * (gripper_goal - start_gripper)
            mujoco.mj_step(model, data)
            peak_height = max(peak_height, float(data.body("task_object").xpos[2]))
            robot_table_hit = False
            object_gripper_hit = False
            for contact in data.contact:
                pair = {int(contact.geom1), int(contact.geom2)}
                if table_id in pair:
                    other = next(iter(pair - {table_id})) if len(pair) == 2 else -1
                    if other >= 0 and int(model.geom_bodyid[other]) in robot_bodies:
                        robot_table_hit = True
                if object_geom in pair:
                    other = next(iter(pair - {object_geom})) if len(pair) == 2 else -1
                    if other >= 0:
                        body = int(model.geom_bodyid[other])
                        if body in (model.body("GRIPPER_LEFT").id, model.body("GRIPPER_RIGH").id):
                            object_gripper_hit = True
            robot_table_steps += int(robot_table_hit)
            object_gripper_steps += int(object_gripper_hit)
            minimum_visual_table_clearance = min(
                minimum_visual_table_clearance,
                float(mujoco.mj_geomDistance(model, data, left_mesh_visual, table_id, 0.2, None)),
                float(mujoco.mj_geomDistance(model, data, right_mesh_visual, table_id, 0.2, None)),
            )
            if closed and object_gripper_hit:
                left_gap = float(mujoco.mj_geomDistance(
                    model, data, left_visual, object_geom, 0.02, None,
                ))
                right_gap = float(mujoco.mj_geomDistance(
                    model, data, right_visual, object_geom, 0.02, None,
                ))
                two_sided_gap = max(left_gap, right_gap)
                closest_visual_gap = min(closest_visual_gap, two_sided_gap)
                visual_contact_steps += int(two_sided_gap <= 0.0)
            if frame_callback is not None:
                frame_callback(model, data, stage)

    final = data.body("task_object").xpos.copy()
    xy_error = float(np.linalg.norm(final[:2] - TARGET_XY))
    target_height = TABLE_TOP_Z + 0.03
    height_error = abs(float(final[2]) - target_height)
    lift_height = peak_height - target_height
    released = bool(abs(data.qpos[6]) < 0.003 and abs(data.qpos[7]) < 0.003)
    success = bool(
        xy_error <= 0.025
        and height_error <= 0.015
        and lift_height >= 0.08
        and robot_table_steps == 0
        and object_gripper_steps > 0
        and visual_contact_steps > 0
        and closest_visual_gap >= -0.003
        and minimum_visual_table_clearance >= 0.0
        and released
    )
    return DrokPickPlaceResult(
        final_xy_error_m=xy_error,
        final_height_error_m=height_error,
        peak_object_height_m=peak_height,
        lift_height_m=lift_height,
        robot_table_contact_steps=robot_table_steps,
        object_gripper_contact_steps=object_gripper_steps,
        visual_contact_steps=visual_contact_steps,
        closest_two_sided_visual_gap_m=closest_visual_gap,
        minimum_visible_gripper_table_clearance_m=minimum_visual_table_clearance,
        final_left_finger_m=float(data.qpos[6]),
        final_right_finger_m=float(data.qpos[7]),
        released=released,
        success=success,
    )


def render_pick_place(output: Path, root: Path = DEFAULT_DROK_ROOT) -> DrokPickPlaceResult:
    import mujoco

    width, height, fps = 960, 540, 30
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        _ffmpeg_executable(), "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)
    assert process.stdin is not None
    renderer = None
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    next_frame = 0.0

    def callback(model, data, stage):
        nonlocal renderer, next_frame
        if renderer is None:
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
            renderer = mujoco.Renderer(model, width=width, height=height)
        if data.time + 1e-9 < next_frame:
            return
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.3, -0.05, 1.0]
        camera.distance = 1.45
        camera.azimuth = 140
        camera.elevation = -18
        renderer.update_scene(data, camera=camera, scene_option=option)
        image = renderer.render().copy()
        image[:8] = (45, 205, 105)
        process.stdin.write(image.tobytes())
        next_frame += 1.0 / fps

    result = run_pick_place(root, callback)
    if renderer is not None:
        renderer.close()
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("DROK ARM 영상 인코딩 실패")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="FDB 과제의 DROK ARM 구조 분석과 pick-place")
    parser.add_argument("--root", type=Path, default=DEFAULT_DROK_ROOT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_drok_pick_place.mp4"))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    structure = analyze_structure(args.root)
    result = run_pick_place(args.root) if args.no_video else render_pick_place(args.output, args.root)
    payload = {
        "source": "https://github.com/jhj0129/DROK_ARM_Sim_only",
        "structure": asdict(structure),
        "runtime_repairs": [
            "ARM_BASE_LINK world Z를 문서 계약값 1.0m로 교정",
            "선언된 stroke 전에 발생하는 손가락끼리의 false mesh contact 제외",
            "원본 은색 외형 내부에 보이지 않는 접촉 전용 충돌 분해 형상 적용",
            "테이블과 물체 접촉 강성을 실제 고체에 가깝게 교정",
        ],
        "result": asdict(result),
        "video": None if args.no_video else str(args.output),
    }
    metrics = args.output.with_suffix(".json")
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
