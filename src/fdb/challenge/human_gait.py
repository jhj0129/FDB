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
    hip_roll: tuple[int, int] | None = None
    hip_roll_amplitude: float = 0.0


GAIT_MAPS = {
    "unitree_g1": GaitMap((0, 6), (3, 9), (4, 10), (15, 22), ("left_ankle_roll_link", "right_ankle_roll_link"), 0.04, 0.16, hip_roll=(1, 7), hip_roll_amplitude=0.08),
    "booster_t1": GaitMap((11, 17), (14, 20), (15, 21), (2, 6), ("left_foot_link", "right_foot_link"), 0.08, 0.20),
    "robotis_op3": GaitMap((10, 16), (11, 17), (12, 18), (2, 5), ("l_ank_roll_link", "r_ank_roll_link"), 0.12, 0.28, -1.0),
    "berkeley_humanoid": GaitMap((2, 8), (3, 9), (4, 10), None, ("ll_faa", "lr_faa"), 0.05, 0.05),
}

PITCH_FEEDBACK_GAINS = {
    "unitree_g1": 0.15,
    "booster_t1": 1.0,
    "robotis_op3": 0.28,
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
    minimum_com_support_margin_m: float | None
    single_support_fraction: float
    double_support_fraction: float
    no_support_fraction: float
    support_phase_transition_count: int
    arm_swing_available: bool
    arm_swing_amplitude_rad: float
    ankle_pitch_feedback_gain: float
    fallen: bool
    upright_complete: bool
    success: bool


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _support_snapshot(model: object, data: object, foot_ids: tuple[int, int]) -> tuple[int, float | None]:
    active: set[int] = set()
    points: list[tuple[float, float]] = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom_ids = (int(contact.geom1), int(contact.geom2))
        body_ids = tuple(int(model.geom_bodyid[geom_id]) for geom_id in geom_ids)
        for side, foot_id in enumerate(foot_ids):
            if foot_id in body_ids and 0 in body_ids:
                active.add(side)
                points.append((float(contact.pos[0]), float(contact.pos[1])))
    hull = _convex_hull(points)
    if len(hull) < 3:
        return len(active), None
    total_mass = float(np.sum(model.body_mass))
    com = np.sum(model.body_mass[:, None] * data.xpos, axis=0) / max(total_mass, 1e-9)
    margins = []
    for start, end in zip(hull, hull[1:] + hull[:1]):
        edge_x, edge_y = end[0] - start[0], end[1] - start[1]
        length = math.hypot(edge_x, edge_y)
        margins.append((edge_x * (float(com[1]) - start[1]) - edge_y * (float(com[0]) - start[0])) / max(length, 1e-9))
    return len(active), min(margins)


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
    if gait.hip_roll is not None:
        shift = ramp * gait.hip_roll_amplitude * wave
        data.ctrl[gait.hip_roll[0]] += shift
        data.ctrl[gait.hip_roll[1]] += shift
    if gait.shoulder_pitch is not None:
        # 사람처럼 반대쪽 다리와 팔을 함께 전진시켜 몸통의 yaw 운동량을 상쇄한다.
        data.ctrl[gait.shoulder_pitch[0]] -= arm_scale * ramp * 0.22 * wave
        data.ctrl[gait.shoulder_pitch[1]] += arm_scale * ramp * 0.22 * wave
    data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def simulate_gait(
    robot_name: str,
    variant: str = "human_like",
    duration_s: float = 4.0,
    floor_friction_scale: float = 1.0,
    push_xy_fraction: tuple[float, float] = (0.0, 0.0),
    step_callback: Callable[[object, object, float], None] | None = None,
) -> tuple[GaitResult, object, object]:
    import mujoco
    import mujoco_menagerie as menagerie

    if variant not in {"human_like", "pitch_feedback", "static_arms", "low_clearance"}:
        raise ValueError(f"unknown gait variant: {variant}")
    model = menagerie.get(robot_name).model("scene")
    plane_ids = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    for plane_id in plane_ids:
        model.geom_friction[plane_id, 0] *= floor_friction_scale
    data = _reset(model)
    gait = GAIT_MAPS[robot_name]
    root_id = _root_body_id(model)
    initial = data.xpos[root_id].copy()
    initial_height = float(initial[2])
    base = data.ctrl.copy()
    feet = [model.body(name).id for name in gait.foot_bodies]
    foot_z = [[float(data.xpos[body_id, 2])] for body_id in feet]
    support_counts = [0, 0, 0]
    support_phase_transition_count = 0
    last_support: int | None = None
    support_margins: list[float] = []
    minimum_height_ratio = 1.0
    minimum_upright = 1.0
    fallen = False
    completed = 0.0
    for step in range(max(1, int(duration_s / model.opt.timestep))):
        time_s = step * model.opt.timestep
        data.xfrc_applied[:] = 0.0
        if 2.0 <= time_s < 2.15:
            total_mass = float(np.sum(model.body_mass))
            data.xfrc_applied[root_id, 0] = push_xy_fraction[0] * total_mass * 9.81
            data.xfrc_applied[root_id, 1] = push_xy_fraction[1] * total_mass * 9.81
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
        if time_s >= 0.5:
            support, margin = _support_snapshot(model, data, (feet[0], feet[1]))
            support_counts[min(2, support)] += 1
            if margin is not None:
                support_margins.append(margin)
            if last_support is not None and support != last_support:
                support_phase_transition_count += 1
            last_support = support
        if height_ratio < 0.55 or upright < 0.5:
            fallen = True
            break
    displacement = data.xpos[root_id] - initial
    distance = math.hypot(float(displacement[0]), float(displacement[1]))
    upright_complete = not fallen and completed >= duration_s - model.opt.timestep
    foot_excursions = (max(foot_z[0]) - min(foot_z[0]), max(foot_z[1]) - min(foot_z[1]))
    support_total = max(1, sum(support_counts))
    single_support_fraction = support_counts[1] / support_total
    success = (
        upright_complete and distance >= 0.01 and max(foot_excursions) >= 0.005
        and single_support_fraction >= 0.045
    )
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
        minimum_com_support_margin_m=min(support_margins) if support_margins else None,
        single_support_fraction=single_support_fraction,
        double_support_fraction=support_counts[2] / support_total,
        no_support_fraction=support_counts[0] / support_total,
        support_phase_transition_count=support_phase_transition_count,
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
        "success_definition": "4초 완주, 낙상 없음, 수평 이동거리 10mm 이상, 한쪽 발 수직 변위 5mm 이상, 단일 지지 4.5% 이상",
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
