from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Callable

import numpy as np

from fdb.challenge.humanoid_suite import DEFAULT_ROBOTS, _reset, _root_body_id
from fdb.v2.render_final import _ffmpeg_executable


@dataclass(frozen=True)
class GaitMap:
    hip_pitch: tuple[int, int]
    knee: tuple[int, int]
    ankle_pitch: tuple[int, int]
    shoulder_pitch: tuple[int, int] | None
    foot_bodies: tuple[str, str]
    hip_amplitude: float
    knee_amplitude: float
    knee_sign: float = 1.0


GAIT_MAPS = {
    "unitree_g1": GaitMap((0, 6), (3, 9), (4, 10), (15, 22), ("left_ankle_roll_link", "right_ankle_roll_link"), 0.06, 0.05),
    "booster_t1": GaitMap((11, 17), (14, 20), (15, 21), (2, 6), ("left_foot_link", "right_foot_link"), 0.08, 0.20),
    "robotis_op3": GaitMap((10, 16), (11, 17), (12, 18), (2, 5), ("l_ank_roll_link", "r_ank_roll_link"), 0.12, 0.28, -1.0),
    "berkeley_humanoid": GaitMap((2, 8), (3, 9), (4, 10), None, ("ll_faa", "lr_faa"), 0.05, 0.05),
}

PITCH_FEEDBACK_GAINS = {
    "unitree_g1": 0.2,
    "booster_t1": 1.0,
    "robotis_op3": 0.2,
    "berkeley_humanoid": 1.0,
}


@dataclass(frozen=True)
class GaitResult:
    robot: str
    variant: str
    completed_duration_s: float
    displacement_xy_m: float
    signed_x_m: float
    lateral_drift_m: float
    average_speed_mps: float
    minimum_height_ratio: float
    minimum_upright_cosine: float
    left_foot_vertical_excursion_m: float
    right_foot_vertical_excursion_m: float
    arm_swing_available: bool
    arm_swing_amplitude_rad: float
    ankle_pitch_feedback_gain: float
    fallen: bool
    upright_complete: bool
    success: bool


def _apply_gait(
    data: object,
    model: object,
    gait: GaitMap,
    base: np.ndarray,
    time_s: float,
    variant: str,
) -> None:
    ramp = min(1.0, max(0.0, (time_s - 0.5) / 0.8))
    phase = 2.0 * math.pi * 0.7 * (time_s - 0.5)
    wave = math.sin(phase)
    left_swing = max(0.0, wave)
    right_swing = max(0.0, -wave)
    knee_scale = 0.25 if variant == "low_clearance" else 1.0
    arm_scale = 0.0 if variant == "static_arms" else 1.0
    hip = gait.hip_amplitude
    knee = gait.knee_amplitude * knee_scale
    data.ctrl[:] = base
    data.ctrl[gait.hip_pitch[0]] += ramp * hip * wave
    data.ctrl[gait.hip_pitch[1]] -= ramp * hip * wave
    data.ctrl[gait.knee[0]] += gait.knee_sign * ramp * knee * left_swing
    data.ctrl[gait.knee[1]] += gait.knee_sign * ramp * knee * right_swing
    data.ctrl[gait.ankle_pitch[0]] -= ramp * hip * 0.55 * wave + gait.knee_sign * ramp * knee * 0.4 * left_swing
    data.ctrl[gait.ankle_pitch[1]] += ramp * hip * 0.55 * wave - gait.knee_sign * ramp * knee * 0.4 * right_swing
    if gait.shoulder_pitch is not None:
        # 사람처럼 반대쪽 다리와 팔을 함께 전진시켜 몸통의 yaw 운동량을 상쇄한다.
        data.ctrl[gait.shoulder_pitch[0]] -= arm_scale * ramp * 0.22 * wave
        data.ctrl[gait.shoulder_pitch[1]] += arm_scale * ramp * 0.22 * wave
    data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def simulate_gait(
    robot_name: str,
    variant: str = "human_like",
    duration_s: float = 4.0,
    step_callback: Callable[[object, object, float], None] | None = None,
) -> tuple[GaitResult, object, object]:
    import mujoco
    import mujoco_menagerie as menagerie

    if variant not in {"human_like", "pitch_feedback", "static_arms", "low_clearance"}:
        raise ValueError(f"unknown gait variant: {variant}")
    model = menagerie.get(robot_name).model("scene")
    data = _reset(model)
    gait = GAIT_MAPS[robot_name]
    root_id = _root_body_id(model)
    initial = data.xpos[root_id].copy()
    initial_height = float(initial[2])
    base = data.ctrl.copy()
    feet = [model.body(name).id for name in gait.foot_bodies]
    foot_z = [[float(data.xpos[body_id, 2])] for body_id in feet]
    minimum_height_ratio = 1.0
    minimum_upright = 1.0
    fallen = False
    completed = 0.0
    for step in range(max(1, int(duration_s / model.opt.timestep))):
        time_s = step * model.opt.timestep
        _apply_gait(data, model, gait, base, time_s, variant)
        if variant == "pitch_feedback":
            root_rotation = data.xmat[root_id].reshape(3, 3)
            gain = PITCH_FEEDBACK_GAINS[robot_name]
            correction = gain * (
                float(root_rotation[0, 2]) + 0.05 * float(data.qvel[4])
            )
            data.ctrl[gait.ankle_pitch[0]] += correction
            data.ctrl[gait.ankle_pitch[1]] += correction
            data.ctrl[:] = np.clip(
                data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
        mujoco.mj_step(model, data)
        completed = (step + 1) * model.opt.timestep
        for values, body_id in zip(foot_z, feet):
            values.append(float(data.xpos[body_id, 2]))
        height_ratio = float(data.xpos[root_id, 2]) / max(initial_height, 1e-6)
        upright = float(data.xmat[root_id].reshape(3, 3)[2, 2])
        minimum_height_ratio = min(minimum_height_ratio, height_ratio)
        minimum_upright = min(minimum_upright, upright)
        if step_callback is not None:
            step_callback(model, data, time_s)
        if height_ratio < 0.55 or upright < 0.5:
            fallen = True
            break
    displacement = data.xpos[root_id] - initial
    distance = math.hypot(float(displacement[0]), float(displacement[1]))
    upright_complete = not fallen and completed >= duration_s - model.opt.timestep
    foot_excursions = (max(foot_z[0]) - min(foot_z[0]), max(foot_z[1]) - min(foot_z[1]))
    success = upright_complete and distance >= 0.01 and max(foot_excursions) >= 0.005
    result = GaitResult(
        robot=robot_name,
        variant=variant,
        completed_duration_s=completed,
        displacement_xy_m=distance,
        signed_x_m=float(displacement[0]),
        lateral_drift_m=float(displacement[1]),
        average_speed_mps=distance / max(completed, 1e-6),
        minimum_height_ratio=minimum_height_ratio,
        minimum_upright_cosine=minimum_upright,
        left_foot_vertical_excursion_m=foot_excursions[0],
        right_foot_vertical_excursion_m=foot_excursions[1],
        arm_swing_available=gait.shoulder_pitch is not None,
        arm_swing_amplitude_rad=0.22 if gait.shoulder_pitch is not None and variant != "static_arms" else 0.0,
        ankle_pitch_feedback_gain=PITCH_FEEDBACK_GAINS[robot_name] if variant == "pitch_feedback" else 0.0,
        fallen=fallen,
        upright_complete=upright_complete,
        success=success,
    )
    return result, model, data


def run_gait_experiment() -> dict[str, object]:
    results = []
    for robot in DEFAULT_ROBOTS:
        for variant in ("human_like", "pitch_feedback", "static_arms", "low_clearance"):
            result, _, _ = simulate_gait(robot, variant)
            results.append(result)
    human_like = [item for item in results if item.variant == "human_like"]
    stabilized = [item for item in results if item.variant == "pitch_feedback"]
    return {
        "schema_version": "1.0",
        "experiment": "human-inspired multi-humanoid gait pilot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robots": list(DEFAULT_ROBOTS),
        "variants": ["human_like", "pitch_feedback", "static_arms", "low_clearance"],
        "success_definition": "4초 완주, 낙상 없음, 수평 이동거리 10mm 이상, 한쪽 발 수직 변위 5mm 이상",
        "human_like_success_count": sum(item.success for item in human_like),
        "human_like_robot_count": len(human_like),
        "pitch_feedback_success_count": sum(item.success for item in stabilized),
        "pitch_feedback_upright_completion_count": sum(item.upright_complete for item in stabilized),
        "results": [asdict(item) for item in results],
    }


def render(output: Path, width: int = 960, height: int = 540, fps: int = 30) -> dict[str, object]:
    import mujoco
    import mujoco_menagerie as menagerie

    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        _ffmpeg_executable(), "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-crf", "21",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)
    assert process.stdin is not None
    panel_width, panel_height = width // 2, height // 2
    states = []
    for robot in DEFAULT_ROBOTS:
        model = menagerie.get(robot).model("scene")
        data = _reset(model)
        root_id = _root_body_id(model)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.0, 0.0, max(0.12, float(data.xpos[root_id, 2]) * 0.55)]
        camera.distance = max(0.85, float(data.xpos[root_id, 2]) * 3.2)
        camera.azimuth = 135
        camera.elevation = -15
        states.append({
            "robot": robot, "model": model, "data": data, "root": root_id,
            "initial": data.xpos[root_id].copy(), "initial_height": float(data.xpos[root_id, 2]),
            "base": data.ctrl.copy(), "camera": camera,
            "renderer": mujoco.Renderer(model, height=panel_height, width=panel_width),
            "fallen": False, "fall_time": None,
        })

    colors = ((40, 190, 255), (255, 170, 40), (80, 230, 110), (230, 80, 100))
    next_frame = 0.0
    total_steps = int(4.0 / 0.002)
    try:
        for step in range(total_steps):
            time_s = step * 0.002
            for state in states:
                model, data = state["model"], state["data"]
                if not state["fallen"]:
                    gait = GAIT_MAPS[state["robot"]]
                    _apply_gait(data, model, gait, state["base"], time_s, "pitch_feedback")
                    rotation = data.xmat[state["root"]].reshape(3, 3)
                    gain = PITCH_FEEDBACK_GAINS[state["robot"]]
                    correction = gain * (float(rotation[0, 2]) + 0.05 * float(data.qvel[4]))
                    data.ctrl[gait.ankle_pitch[0]] += correction
                    data.ctrl[gait.ankle_pitch[1]] += correction
                    data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
                mujoco.mj_step(model, data)
                ratio = float(data.xpos[state["root"], 2]) / state["initial_height"]
                upright = float(data.xmat[state["root"]].reshape(3, 3)[2, 2])
                if not state["fallen"] and (ratio < 0.55 or upright < 0.5):
                    state["fallen"] = True
                    state["fall_time"] = time_s
            if time_s + 1e-9 >= next_frame:
                panels = []
                for index, state in enumerate(states):
                    state["renderer"].update_scene(state["data"], camera=state["camera"])
                    panel = state["renderer"].render().copy()
                    panel[:5, :, :] = colors[index]
                    panel[-5:, :, :] = colors[index]
                    panel[:, :5, :] = colors[index]
                    panel[:, -5:, :] = colors[index]
                    panels.append(panel)
                process.stdin.write(np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))).tobytes())
                next_frame += 1.0 / fps
    finally:
        for state in states:
            state["renderer"].close()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("보행 비교 영상 인코딩 실패")
    return {
        "video": str(output), "duration_s": 4.0, "fps": fps, "resolution": [width, height],
        "layout": {"top_left_blue": "unitree_g1", "top_right_orange": "booster_t1", "bottom_left_green": "robotis_op3", "bottom_right_red": "berkeley_humanoid"},
        "controller": "반대 위상 다리, 유각기 무릎 굽힘, 반대쪽 팔 스윙, 몸통 pitch 기반 양 발목 폐루프 보상",
        "final_states": {
            state["robot"]: {
                "displacement_xy_m": math.hypot(float(state["data"].xpos[state["root"], 0] - state["initial"][0]), float(state["data"].xpos[state["root"], 1] - state["initial"][1])),
                "fallen": state["fallen"], "fall_time_s": state["fall_time"],
            } for state in states
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="사람 보행 원리를 적용한 다중 휴머노이드 실험")
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_human_gait.mp4"))
    args = parser.parse_args()
    metrics = run_gait_experiment()
    video = render(args.output)
    combined = {**metrics, "render": video}
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frame_path = args.output.with_suffix(".png")
    subprocess.run([_ffmpeg_executable(), "-loglevel", "error", "-y", "-ss", "1.8", "-i", str(args.output), "-frames:v", "1", str(frame_path)], check=True)
    print(json.dumps({**combined, "metrics": str(metrics_path), "preview": str(frame_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
