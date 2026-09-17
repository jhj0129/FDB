from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Callable

import numpy as np

from fdb.v2.inspector import load_robot
from fdb.v2.pose_reach import PandaPoseReachExperiment


@dataclass(frozen=True)
class SortObject:
    object_id: str
    shape: str
    color: str
    size_class: str
    source_xy: tuple[float, float]
    target_xy: tuple[float, float]
    half_size: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


OBJECTS = (
    SortObject("red_cube", "box", "red", "small", (0.40, 0.17), (0.55, -0.16), (0.018, 0.018, 0.022), (0.95, 0.08, 0.04, 1.0)),
    SortObject("green_wide", "box", "green", "wide", (0.49, 0.17), (0.58, 0.00), (0.023, 0.017, 0.022), (0.08, 0.85, 0.15, 1.0)),
    SortObject("blue_tall", "box", "blue", "tall", (0.58, 0.17), (0.55, 0.16), (0.017, 0.017, 0.029), (0.08, 0.25, 0.95, 1.0)),
)


@dataclass(frozen=True)
class ObjectSortResult:
    object_id: str
    predicted_class: str
    expected_class: str
    classification_correct: bool
    final_xy_error_m: float
    final_z_error_m: float
    lifted_m: float
    released: bool
    success: bool


def classify_object(obj: SortObject) -> str:
    """Deterministic perception baseline from observable color and dimensions."""
    if obj.color == "red" and max(obj.half_size[:2]) <= 0.020:
        return "red_small"
    if obj.color == "green" and obj.half_size[0] > obj.half_size[1]:
        return "green_wide"
    if obj.color == "blue" and obj.half_size[2] > max(obj.half_size[:2]):
        return "blue_tall"
    return "unknown"


def _expected_class(obj: SortObject) -> str:
    return f"{obj.color}_{obj.size_class}"


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _build_scene():
    import mujoco

    record, _ = load_robot()
    spec = record.spec(record.default_model)
    spec.worldbody.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.5, 0.0, 0.29],
        size=[0.4, 0.4, 0.02], friction=[1.0, 0.005, 0.0001],
        rgba=[0.38, 0.29, 0.21, 1.0],
    )
    zone_colors = ((0.95, 0.08, 0.04, 0.55), (0.08, 0.85, 0.15, 0.55), (0.08, 0.25, 0.95, 0.55))
    for obj, color in zip(OBJECTS, zone_colors):
        spec.worldbody.add_geom(
            name=f"zone_{obj.object_id}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[*obj.target_xy, 0.312], size=[0.045, 0.002, 0.0],
            contype=0, conaffinity=0, rgba=color,
        )
        body = spec.worldbody.add_body(
            name=obj.object_id,
            pos=[*obj.source_xy, 0.31 + obj.half_size[2]],
        )
        body.add_freejoint(name=f"{obj.object_id}_free")
        body.add_geom(
            name=f"{obj.object_id}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=obj.half_size, density=300.0,
            friction=[2.0, 0.01, 0.001], rgba=obj.rgba,
        )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:9] = model.key(0).qpos[:9]
    data.ctrl[:] = model.key(0).ctrl
    mujoco.mj_forward(model, data)
    return model, data


def run_sorting(
    frame_callback: Callable[[object, object, str], None] | None = None,
) -> dict[str, object]:
    import mujoco

    model, data = _build_scene()
    table_id = model.geom("table").id
    object_geom_ids = {model.geom(f"{obj.object_id}_geom").id for obj in OBJECTS}
    robot_table_contact_steps = 0
    deepest_penetration_m = 0.0
    minimum_hand_clearance_m = math.inf
    peaks = {obj.object_id: float(data.body(obj.object_id).xpos[2]) for obj in OBJECTS}

    for obj in OBJECTS:
        sx, sy = obj.source_xy
        tx, ty = obj.target_xy
        grasp_z = 0.31 + obj.half_size[2] + 0.10
        poses = (
            (sx, sy, 0.54), (sx, sy, grasp_z), (sx, sy, grasp_z + 0.20),
            (tx, ty, grasp_z + 0.20), (tx, ty, 0.455), (tx, ty, 0.54),
        )
        targets = [PandaPoseReachExperiment(pose).run()[0].target_qpos for pose in poses]
        stages = (
            ("pregrasp", targets[0], 255.0, 700),
            ("descend", targets[1], 255.0, 650),
            ("grasp", targets[1], 0.0, 650),
            ("lift", targets[2], 0.0, 800),
            ("transfer", targets[3], 0.0, 900),
            ("lower", targets[4], 0.0, 750),
            ("release", targets[4], 255.0, 600),
            ("retreat", targets[5], 255.0, 600),
        )
        for stage_name, target, gripper, steps in stages:
            start_arm = data.ctrl[:7].copy()
            start_gripper = float(data.ctrl[7])
            for step in range(steps):
                phase = (step + 1) / steps
                blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
                data.ctrl[:7] = start_arm + blend * (target - start_arm)
                data.ctrl[7] = start_gripper + blend * (gripper - start_gripper)
                mujoco.mj_step(model, data)
                minimum_hand_clearance_m = min(
                    minimum_hand_clearance_m,
                    float(data.body("hand").xpos[2]) - 0.31,
                )
                for tracked in OBJECTS:
                    peaks[tracked.object_id] = max(
                        peaks[tracked.object_id], float(data.body(tracked.object_id).xpos[2])
                    )
                contacted = False
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if table_id in pair and pair.isdisjoint(object_geom_ids):
                        contacted = True
                        deepest_penetration_m = max(deepest_penetration_m, max(0.0, -float(contact.dist)))
                if contacted:
                    robot_table_contact_steps += 1
                if frame_callback is not None:
                    frame_callback(model, data, f"{obj.object_id}:{stage_name}")

    results = []
    for obj in OBJECTS:
        final = data.body(obj.object_id).xpos
        target_z = 0.31 + obj.half_size[2]
        xy_error = math.hypot(float(final[0]) - obj.target_xy[0], float(final[1]) - obj.target_xy[1])
        z_error = abs(float(final[2]) - target_z)
        predicted = classify_object(obj)
        expected = _expected_class(obj)
        released = float(data.joint("finger_joint1").qpos[0]) >= 0.035
        lifted = peaks[obj.object_id] - target_z
        success = predicted == expected and xy_error <= 0.055 and z_error <= 0.022 and lifted >= 0.07 and released
        results.append(ObjectSortResult(
            obj.object_id, predicted, expected, predicted == expected,
            xy_error, z_error, lifted, released, success,
        ))
    success_count = sum(item.success for item in results)
    return {
        "task": "multi_object_classify_pick_sort",
        "object_count": len(OBJECTS),
        "classification_accuracy": sum(item.classification_correct for item in results) / len(results),
        "sorting_success_count": success_count,
        "sorting_success_rate": success_count / len(results),
        "robot_table_contact_steps": robot_table_contact_steps,
        "deepest_robot_table_penetration_m": deepest_penetration_m,
        "minimum_hand_table_clearance_m": minimum_hand_clearance_m,
        "hard_safety_pass": robot_table_contact_steps == 0 and deepest_penetration_m == 0.0 and minimum_hand_clearance_m >= 0.055,
        "results": [asdict(item) for item in results],
        "model": model,
        "data": data,
    }


def render(output: Path, width: int = 640, height: int = 480, fps: int = 30) -> dict[str, object]:
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
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.49, 0.0, 0.38]
    camera.distance = 1.55
    camera.azimuth = 135
    camera.elevation = -28
    next_time = 0.0

    def capture(model, data, _stage):
        nonlocal renderer, next_time
        if renderer is None:
            renderer = mujoco.Renderer(model, height=height, width=width)
        if data.time + 1e-9 >= next_time:
            renderer.update_scene(data, camera=camera)
            process.stdin.write(renderer.render().tobytes())
            next_time += 1.0 / fps

    metrics = run_sorting(capture)
    if renderer is not None:
        for _ in range(fps):
            renderer.update_scene(metrics["data"], camera=camera)
            process.stdin.write(renderer.render().tobytes())
        renderer.close()
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("다중 물체 분류 영상 인코딩 실패")
    metrics.pop("model")
    metrics.pop("data")
    metrics["video"] = str(output)
    metrics["fps"] = fps
    metrics["resolution"] = [width, height]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Panda 다중 물체 분류·배치 실험")
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_multi_object_sorting.mp4"))
    args = parser.parse_args()
    metrics = render(args.output)
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frame_path = args.output.with_suffix(".png")
    subprocess.run([_ffmpeg_executable(), "-loglevel", "error", "-y", "-sseof", "-0.15", "-i", str(args.output), "-frames:v", "1", str(frame_path)], check=True)
    print(json.dumps({**metrics, "metrics": str(metrics_path), "final_frame": str(frame_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
