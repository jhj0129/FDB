from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from fdb.challenge.drok_arm_tasks import (
    DEFAULT_DROK_ROOT,
    TABLE_TOP_Z,
    _solve_pose,
    repaired_robot_spec,
)
from fdb.challenge.multi_object_sorting import (
    OBJECTS,
    SortObject,
    _expected_class,
    classify_object_neural,
)
from fdb.v2.render_final import _ffmpeg_executable


DROK_OBJECTS = (
    replace(OBJECTS[0], source_xy=(0.38, 0.13), target_xy=(0.40, -0.18)),
    replace(OBJECTS[1], source_xy=(0.48, 0.13), target_xy=(0.49, -0.18)),
    replace(OBJECTS[2], source_xy=(0.58, 0.13), target_xy=(0.58, -0.18)),
)


@dataclass(frozen=True)
class DrokSortResult:
    object_id: str
    predicted_class: str
    expected_class: str
    classification_confidence: float
    classification_correct: bool
    final_xy_error_m: float
    final_z_error_m: float
    lifted_m: float
    gripper_contact_steps: int
    released: bool
    success: bool


def build_sorting_scene(
    root: Path = DEFAULT_DROK_ROOT,
    objects: tuple[SortObject, ...] = DROK_OBJECTS,
):
    import mujoco

    spec = repaired_robot_spec(root)
    spec.worldbody.add_geom(
        name="work_table", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.48, -0.03, TABLE_TOP_Z - 0.04], size=[0.34, 0.36, 0.04],
        rgba=[0.36, 0.24, 0.13, 1.0], friction=[1.2, 0.01, 0.001],
        solref=[0.003, 1.0], group=5,
    )
    for obj in objects:
        spec.worldbody.add_geom(
            name=f"zone_{obj.object_id}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[*obj.target_xy, TABLE_TOP_Z + 0.001], size=[0.038, 0.001],
            rgba=[*obj.rgba[:3], 0.55], contype=0, conaffinity=0, group=5,
        )
        body = spec.worldbody.add_body(
            name=obj.object_id,
            pos=[*obj.source_xy, TABLE_TOP_Z + obj.half_size[2]],
        )
        body.add_freejoint(name=f"{obj.object_id}_free")
        body.add_geom(
            name=f"{obj.object_id}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=obj.half_size, density=500.0,
            friction=[5.0, 0.05, 0.005], solref=[0.003, 1.0],
            rgba=obj.rgba, group=5,
        )
    return spec.compile()


def _object_waypoints(model, obj: SortObject, seed: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    top_down = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    positions = (
        np.r_[obj.source_xy, 1.15],
        np.r_[obj.source_xy, 0.968],
        np.r_[obj.source_xy, 1.15],
        np.r_[obj.target_xy, 1.15],
        np.r_[obj.target_xy, 0.968],
        np.r_[obj.target_xy, 1.15],
    )
    waypoints: list[np.ndarray] = []
    current = seed
    for position in positions:
        current = _solve_pose(model, position, top_down, current)
        waypoints.append(current)
    return waypoints, current


def run_sorting(
    root: Path = DEFAULT_DROK_ROOT,
    objects: tuple[SortObject, ...] = DROK_OBJECTS,
    frame_callback: Callable[[object, object, str], None] | None = None,
) -> dict[str, object]:
    import mujoco

    model = build_sorting_scene(root, objects)
    plans: list[tuple[SortObject, list[np.ndarray]]] = []
    seed = np.asarray([0.0, 1.45, 0.0, -1.45, 1.20, -1.57])
    for obj in objects:
        waypoints, seed = _object_waypoints(model, obj, seed)
        plans.append((obj, waypoints))

    data = mujoco.MjData(model)
    data.qpos[:6] = plans[0][1][0]
    data.qpos[6:8] = 0.0
    data.ctrl[:6] = plans[0][1][0]
    data.ctrl[6:8] = 0.0
    for obj in objects:
        address = model.joint(f"{obj.object_id}_free").qposadr[0]
        data.qpos[address:address + 7] = [
            *obj.source_xy, TABLE_TOP_Z + obj.half_size[2], 1, 0, 0, 0,
        ]
    mujoco.mj_forward(model, data)

    table_id = model.geom("work_table").id
    robot_body_ids = {
        model.body(name).id for name in (
            "ARM_BASE_LINK", "LINK1", "LINK2", "LINK3",
            "LINK4", "LINK5", "LINK6", "GRIPPER_BASE",
            "GRIPPER_LEFT", "GRIPPER_RIGH",
        )
    }
    object_geom_ids = {obj.object_id: model.geom(f"{obj.object_id}_geom").id for obj in objects}
    peaks = {obj.object_id: float(data.body(obj.object_id).xpos[2]) for obj in objects}
    contact_steps = {obj.object_id: 0 for obj in objects}
    robot_table_contact_steps = 0

    for obj, waypoint in plans:
        object_width = 2.0 * obj.half_size[1]
        close_distance = min(0.045, (0.080 - object_width) / 2.0 + 0.0005)
        stages = (
            ("approach", waypoint[0], False, 900),
            ("descend", waypoint[1], False, 1000),
            ("grasp", waypoint[1], True, 1100),
            ("lift", waypoint[2], True, 1300),
            ("classify_transfer", waypoint[3], True, 1500),
            ("place", waypoint[4], True, 1100),
            ("release", waypoint[4], False, 900),
            ("retreat", waypoint[5], False, 1000),
        )
        for stage, target, closed, steps in stages:
            start_arm = data.ctrl[:6].copy()
            start_gripper = data.ctrl[6:8].copy()
            goal = np.asarray([close_distance, -close_distance]) if closed else np.zeros(2)
            for index in range(steps):
                phase = (index + 1) / steps
                blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
                data.ctrl[:6] = start_arm + blend * (target - start_arm)
                data.ctrl[6:8] = start_gripper + blend * (goal - start_gripper)
                mujoco.mj_step(model, data)
                for tracked in objects:
                    peaks[tracked.object_id] = max(
                        peaks[tracked.object_id], float(data.body(tracked.object_id).xpos[2])
                    )
                table_hit = False
                object_hit = False
                for contact in data.contact:
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if table_id in pair:
                        other = next(iter(pair - {table_id})) if len(pair) == 2 else -1
                        if other >= 0 and int(model.geom_bodyid[other]) in robot_body_ids:
                            table_hit = True
                    if object_geom_ids[obj.object_id] in pair:
                        other = next(iter(pair - {object_geom_ids[obj.object_id]})) if len(pair) == 2 else -1
                        if other >= 0 and int(model.geom_bodyid[other]) in {
                            model.body("GRIPPER_LEFT").id, model.body("GRIPPER_RIGH").id,
                        }:
                            object_hit = True
                robot_table_contact_steps += int(table_hit)
                contact_steps[obj.object_id] += int(object_hit)
                if frame_callback is not None:
                    frame_callback(model, data, f"{obj.object_id}:{stage}")

    results: list[DrokSortResult] = []
    for obj in objects:
        final = data.body(obj.object_id).xpos
        target_z = TABLE_TOP_Z + obj.half_size[2]
        xy_error = math.hypot(float(final[0]) - obj.target_xy[0], float(final[1]) - obj.target_xy[1])
        z_error = abs(float(final[2]) - target_z)
        prediction, confidence = classify_object_neural(obj)
        expected = _expected_class(obj)
        lifted = peaks[obj.object_id] - target_z
        released = bool(abs(data.qpos[6]) < 0.003 and abs(data.qpos[7]) < 0.003)
        success = bool(
            prediction == expected and xy_error <= 0.025 and z_error <= 0.012
            and lifted >= 0.08 and contact_steps[obj.object_id] > 0 and released
        )
        results.append(DrokSortResult(
            obj.object_id, prediction, expected, confidence, prediction == expected,
            xy_error, z_error, lifted, contact_steps[obj.object_id], released, success,
        ))
    successes = sum(result.success for result in results)
    return {
        "task": "DROK ARM 신경망 다중 물체 분류·배치",
        "classification_method": "five_member_mlp_ensemble",
        "classification_accuracy": sum(result.classification_correct for result in results) / len(results),
        "sorting_success_count": successes,
        "sorting_success_rate": successes / len(results),
        "robot_table_contact_steps": robot_table_contact_steps,
        "hard_safety_pass": robot_table_contact_steps == 0,
        "results": [asdict(result) for result in results],
        "model": model,
        "data": data,
    }


def render_sorting(
    output: Path,
    root: Path = DEFAULT_DROK_ROOT,
    width: int = 960,
    height: int = 540,
    fps: int = 30,
) -> dict[str, object]:
    import mujoco

    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        _ffmpeg_executable(), "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-crf", "21",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)
    assert process.stdin is not None
    renderer = None
    next_frame = 0.0
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.32, -0.02, 1.03]
    camera.distance = 1.55
    camera.azimuth = 140
    camera.elevation = -22
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 1

    def capture(model, data, _stage):
        nonlocal renderer, next_frame
        if renderer is None:
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
            renderer = mujoco.Renderer(model, width=width, height=height)
        if data.time + 1e-9 < next_frame:
            return
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        process.stdin.write(renderer.render().tobytes())
        next_frame += 1.0 / fps

    metrics = run_sorting(root=root, frame_callback=capture)
    if renderer is not None:
        for _ in range(fps):
            renderer.update_scene(metrics["data"], camera=camera, scene_option=scene_option)
            process.stdin.write(renderer.render().tobytes())
        renderer.close()
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("DROK ARM 분류 영상 인코딩 실패")
    metrics.pop("model")
    metrics.pop("data")
    metrics.update(video=str(output), fps=fps, resolution=[width, height])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="DROK ARM 신경망 다중 물체 분류·배치")
    parser.add_argument("--root", type=Path, default=DEFAULT_DROK_ROOT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_drok_arm_sorting.mp4"))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    metrics = run_sorting(args.root) if args.no_video else render_sorting(args.output, args.root)
    metrics.pop("model", None)
    metrics.pop("data", None)
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.no_video:
        frame = args.output.with_suffix(".png")
        subprocess.run([
            _ffmpeg_executable(), "-loglevel", "error", "-y", "-sseof", "-0.15",
            "-i", str(args.output), "-frames:v", "1", str(frame),
        ], check=True)
        metrics["final_frame"] = str(frame)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
