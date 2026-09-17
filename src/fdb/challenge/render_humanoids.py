from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from fdb.v2.render_final import _ffmpeg_executable
from .humanoid_suite import DEFAULT_ROBOTS, _actuator_match, _reset, _root_body_id


def _motion_targets(robot_name: str, model: object, data: object) -> dict[int, float]:
    if robot_name == "unitree_g1":
        ids = _actuator_match(model, ("left", "shoulder", "pitch"))
        delta = 0.45
    elif robot_name == "booster_t1":
        ids = _actuator_match(model, ("head", "yaw"))
        delta = 0.35
    elif robot_name == "robotis_op3":
        ids = _actuator_match(model, ("l_knee",)) + _actuator_match(model, ("r_knee",))
        delta = 0.35
    else:
        ids = _actuator_match(model, ("ll_kfe",)) + _actuator_match(model, ("lr_kfe",))
        delta = -0.35
    return {
        actuator_id: float(np.clip(
            data.ctrl[actuator_id] + delta,
            model.actuator_ctrlrange[actuator_id, 0],
            model.actuator_ctrlrange[actuator_id, 1],
        ))
        for actuator_id in ids
    }


def render(
    output: Path,
    *,
    width: int = 960,
    height: int = 540,
    fps: int = 30,
    duration_s: float = 6.0,
) -> dict[str, object]:
    import mujoco
    import mujoco_menagerie as menagerie

    panel_width, panel_height = width // 2, height // 2
    states = []
    for robot_name in DEFAULT_ROBOTS:
        source = menagerie.get(robot_name)
        model = source.model("scene")
        data = _reset(model)
        root_id = _root_body_id(model)
        initial_height = float(data.xpos[root_id, 2])
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.0, 0.0, max(0.12, initial_height * 0.55)]
        camera.distance = max(0.85, initial_height * 3.2)
        camera.azimuth = 135
        camera.elevation = -15
        renderer = mujoco.Renderer(model, height=panel_height, width=panel_width)
        targets = _motion_targets(robot_name, model, data)
        starts = {actuator_id: float(data.ctrl[actuator_id]) for actuator_id in targets}
        states.append({
            "name": robot_name,
            "model": model,
            "data": data,
            "root_id": root_id,
            "initial_height": initial_height,
            "camera": camera,
            "renderer": renderer,
            "targets": targets,
            "starts": starts,
            "mass": float(np.sum(model.body_mass)),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            _ffmpeg_executable(), "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-an", "-c:v", "libx264", "-crf", "21",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("영상 인코더 입력을 열 수 없습니다")

    def compose_frame() -> bytes:
        panels = []
        for index, state in enumerate(states):
            state["renderer"].update_scene(state["data"], camera=state["camera"])
            panel = state["renderer"].render().copy()
            color = ((40, 190, 255), (255, 170, 40), (80, 230, 110), (230, 80, 100))[index]
            panel[:5, :, :] = color
            panel[-5:, :, :] = color
            panel[:, :5, :] = color
            panel[:, -5:, :] = color
            panels.append(panel)
        return np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))).tobytes()

    total_steps = int(duration_s / states[0]["model"].opt.timestep)
    next_frame_time = 0.0
    try:
        for step in range(total_steps):
            time_s = step * states[0]["model"].opt.timestep
            for state in states:
                model, data = state["model"], state["data"]
                data.xfrc_applied[:] = 0.0
                if 1.0 <= time_s <= 2.5:
                    phase = min(1.0, (time_s - 1.0) / 1.0)
                    blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
                    for actuator_id, target in state["targets"].items():
                        start = state["starts"][actuator_id]
                        data.ctrl[actuator_id] = start + blend * (target - start)
                if 3.0 <= time_s < 3.15:
                    data.xfrc_applied[state["root_id"], 0] = 0.15 * state["mass"] * 9.81
                mujoco.mj_step(model, data)
            if time_s + 1e-9 >= next_frame_time:
                process.stdin.write(compose_frame())
                next_frame_time += 1.0 / fps
    finally:
        for state in states:
            state["renderer"].close()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("휴머노이드 비교 영상 인코딩 실패")
    final_states = {}
    for state in states:
        model, data = state["model"], state["data"]
        height_ratio = float(data.xpos[state["root_id"], 2]) / max(state["initial_height"], 1e-6)
        upright = float(data.xmat[state["root_id"]].reshape(3, 3)[2, 2])
        final_states[state["name"]] = {
            "height_ratio": height_ratio,
            "upright_cosine": upright,
            "fallen": bool(height_ratio < 0.55 or upright < 0.5),
        }
    return {
        "video": str(output),
        "duration_s": duration_s,
        "fps": fps,
        "resolution": [width, height],
        "layout": {
            "top_left_blue": "unitree_g1",
            "top_right_orange": "booster_t1",
            "bottom_left_green": "robotis_op3",
            "bottom_right_red": "berkeley_humanoid",
        },
        "timeline": {
            "0_to_1s": "stand baseline",
            "1_to_2_5s": "robot-specific arm, head, or knee task",
            "3_to_3_15s": "mass-scaled horizontal push",
            "3_15_to_6s": "recovery observation",
        },
        "final_states": final_states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="네 휴머노이드 비교 영상")
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/fdb_humanoid_challenge.mp4"),
    )
    args = parser.parse_args()
    metrics = render(args.output)
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frame_path = args.output.with_suffix(".png")
    subprocess.run(
        [
            _ffmpeg_executable(), "-loglevel", "error", "-y", "-sseof", "-0.1",
            "-i", str(args.output), "-frames:v", "1", str(frame_path),
        ],
        check=True,
    )
    print(json.dumps({**metrics, "metrics": str(metrics_path), "final_frame": str(frame_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
