from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from .inspector import load_robot
from .pick_place import PandaPickPlaceExperiment
from .pose_reach import PandaPoseReachExperiment


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError("Install the 'robot' extra to render MP4 video") from error


def render(output: Path, width: int = 640, height: int = 360, fps: int = 30) -> dict[str, object]:
    """Render the selected Panda pick-and-place future and return verified metrics."""
    import mujoco

    experiment = PandaPickPlaceExperiment()
    selected, candidates = experiment.run()
    if not selected.success:
        raise RuntimeError("Refusing to render: no pick-and-place candidate succeeded")

    record, _ = load_robot()
    spec = record.spec(record.default_model)
    spec.worldbody.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.5, 0.0, 0.29],
        size=[0.4, 0.4, 0.02], friction=[1.0, 0.005, 0.0001],
        rgba=[0.40, 0.30, 0.22, 1.0],
    )
    spec.worldbody.add_geom(
        name="target_zone", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[*experiment.scenario.target_xy, 0.312], size=[0.06, 0.002, 0.0],
        contype=0, conaffinity=0, rgba=[0.1, 0.35, 1.0, 0.85],
    )
    obj = spec.worldbody.add_body(name="object", pos=experiment.source)
    obj.add_freejoint(name="object_free")
    obj.add_geom(
        name="object_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=experiment.scenario.half_size, density=experiment.scenario.density,
        friction=[experiment.scenario.sliding_friction, 0.01, 0.001],
        rgba=[1.0, 0.08, 0.04, 1.0],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:9] = model.key(0).qpos[:9]
    data.ctrl[:] = model.key(0).ctrl
    mujoco.mj_forward(model, data)

    sx, sy, sz = experiment.source
    tx, ty, tz = experiment.target
    poses = (
        (sx, sy, 0.52),
        (sx, sy, experiment.grasp_hand_z),
        (sx, sy, experiment.grasp_hand_z + 0.20),
        (tx, ty, experiment.grasp_hand_z + 0.20),
        (tx, ty, selected.plan.place_hand_z),
        (tx, ty, 0.52),
    )
    joint_targets = [PandaPoseReachExperiment(pose).run()[0].target_qpos for pose in poses]
    stages = (
        ("pregrasp", joint_targets[0], 255.0, 1200),
        ("descend", joint_targets[1], 255.0, 1200),
        ("grasp", joint_targets[1], 0.0, 1200),
        ("lift", joint_targets[2], 0.0, 1500),
        ("transfer", joint_targets[3], 0.0, 1600),
        ("lower", joint_targets[4], 0.0, 1400),
        ("release", joint_targets[4], 255.0, 1000),
        ("retreat", joint_targets[5], 255.0, 1000),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _ffmpeg_executable(), "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Could not open video encoder input")

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.45, 0.0, 0.38]
    camera.distance = 1.55
    camera.azimuth = 132
    camera.elevation = -24
    renderer = mujoco.Renderer(model, height=height, width=width)
    peak_z = sz
    stable = True
    frame_interval = 1.0 / fps
    next_frame_time = 0.0

    def write_frame() -> None:
        renderer.update_scene(data, camera=camera)
        process.stdin.write(renderer.render().tobytes())

    try:
        write_frame()
        next_frame_time += frame_interval
        for _name, target_qpos, gripper, steps in stages:
            data.ctrl[:7] = target_qpos
            data.ctrl[7] = gripper
            for _ in range(steps):
                mujoco.mj_step(model, data)
                peak_z = max(peak_z, float(data.body("object").xpos[2]))
                if not all(math.isfinite(float(value)) for value in data.qpos):
                    stable = False
                    break
                while data.time >= next_frame_time:
                    write_frame()
                    next_frame_time += frame_interval
            if not stable:
                break
        for _ in range(fps):
            write_frame()
    finally:
        renderer.close()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding the final simulation")

    final = data.body("object").xpos.copy()
    xy_error = math.hypot(float(final[0]) - tx, float(final[1]) - ty)
    released = float(data.joint("finger_joint1").qpos[0]) >= 0.035
    success = (
        stable and peak_z - sz >= 0.08 and xy_error <= 0.05
        and abs(float(final[2]) - tz) <= 0.02 and released
    )
    metrics: dict[str, object] = {
        "selected_plan": selected.plan.plan_id,
        "candidate_results": [asdict(item) for item in candidates],
        "rendered_success": success,
        "pick_lift_height_m": peak_z - sz,
        "final_xy_error_m": xy_error,
        "final_object_z": float(final[2]),
        "released": released,
        "stable": stable,
        "duration_s": float(data.time) + 1.0,
        "fps": fps,
        "resolution": [width, height],
        "video": str(output),
    }
    if not success:
        raise RuntimeError(f"Rendered trajectory failed verification: {metrics}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_final_pick_place.mp4"))
    args = parser.parse_args()
    metrics = render(args.output)
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    frame_path = args.output.with_suffix(".png")
    subprocess.run(
        [_ffmpeg_executable(), "-loglevel", "error", "-y", "-sseof", "-0.2", "-i", str(args.output), "-frames:v", "1", str(frame_path)],
        check=True,
    )
    print(json.dumps({**metrics, "metrics": str(metrics_path), "final_frame": str(frame_path)}, indent=2))


if __name__ == "__main__":
    main()
