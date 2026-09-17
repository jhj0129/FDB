from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess

import numpy as np

from fdb.v2.render_final import _ffmpeg_executable


LEFT_FOOT_SITE = 3
RIGHT_FOOT_SITE = 6


@dataclass(frozen=True)
class Touchdown:
    frame: int
    side: str
    lift_frame: int
    air_time_s: float
    maximum_foot_height_m: float
    forward_placement_m: float


@dataclass(frozen=True)
class BehavioralGaitResult:
    source: str
    robot: str
    start_frame: int
    end_frame: int
    duration_s: float
    alternating_touchdowns: int
    completed_strides: int
    left_touchdowns: int
    right_touchdowns: int
    minimum_swing_clearance_m: float
    minimum_forward_placement_m: float
    pelvis_path_length_m: float
    net_pelvis_displacement_m: float
    success: bool


def detect_touchdowns(
    site_xpos: np.ndarray,
    root_xy: np.ndarray,
    frequency: float,
    lift_height_m: float = 0.080,
    touchdown_height_m: float = 0.055,
    minimum_air_frames: int = 5,
) -> list[Touchdown]:
    """Detect real swing-to-stance events with hysteresis, not foot jitter."""
    events: list[Touchdown] = []
    for side, site_index in (("left", LEFT_FOOT_SITE), ("right", RIGHT_FOOT_SITE)):
        foot = np.asarray(site_xpos[:, site_index], dtype=float)
        state = "ground" if foot[0, 2] < touchdown_height_m else "air"
        lift_frame: int | None = None
        for frame, z_value in enumerate(foot[:, 2]):
            if state == "ground" and z_value > lift_height_m:
                state = "air"
                lift_frame = frame
            elif state == "air" and z_value < touchdown_height_m:
                if lift_frame is not None and frame - lift_frame >= minimum_air_frames:
                    pelvis_delta = root_xy[frame] - root_xy[lift_frame]
                    direction = pelvis_delta / max(float(np.linalg.norm(pelvis_delta)), 1e-9)
                    foot_delta = foot[frame, :2] - foot[lift_frame, :2]
                    events.append(Touchdown(
                        frame=frame,
                        side=side,
                        lift_frame=lift_frame,
                        air_time_s=(frame - lift_frame) / frequency,
                        maximum_foot_height_m=float(np.max(foot[lift_frame:frame + 1, 2])),
                        forward_placement_m=float(foot_delta @ direction),
                    ))
                state = "ground"
                lift_frame = None
    return sorted(events, key=lambda event: event.frame)


def longest_alternating_run(events: list[Touchdown], maximum_gap_frames: int = 80) -> list[Touchdown]:
    best: list[Touchdown] = []
    current: list[Touchdown] = []
    for event in events:
        if not current or (
            event.side != current[-1].side
            and 0 < event.frame - current[-1].frame <= maximum_gap_frames
        ):
            current.append(event)
        else:
            current = [event]
        if len(current) > len(best):
            best = current.copy()
    return best


def best_alternating_window(
    events: list[Touchdown], required_steps: int, maximum_gap_frames: int
) -> list[Touchdown]:
    candidates: list[list[Touchdown]] = []
    for start in range(max(0, len(events) - required_steps + 1)):
        window = events[start:start + required_steps]
        if len(window) < required_steps:
            continue
        if all(
            current.side != following.side
            and 0 < following.frame - current.frame <= maximum_gap_frames
            for current, following in zip(window, window[1:])
        ):
            candidates.append(window)
    if not candidates:
        return []
    # 가장 약한 한 걸음이 좋은 구간을 택한다. 평균이 좋아도 한 발이 끌리면 탈락한다.
    return max(candidates, key=lambda window: (
        min(event.forward_placement_m for event in window),
        min(event.maximum_foot_height_m for event in window),
    ))


def evaluate_behavioral_walk(
    site_xpos: np.ndarray,
    root_xy: np.ndarray,
    frequency: float,
    required_steps: int = 10,
) -> tuple[BehavioralGaitResult, list[Touchdown]]:
    events = detect_touchdowns(site_xpos, root_xy, frequency)
    selected = best_alternating_window(events, required_steps, round(1.6 * frequency))
    if not selected:
        raise ValueError("연속 보행 접지 이벤트를 찾지 못했습니다.")
    start = max(0, selected[0].lift_frame - round(0.4 * frequency))
    end = min(len(root_xy) - 1, selected[-1].frame + round(0.4 * frequency))
    deltas = np.diff(root_xy[start:end + 1], axis=0)
    path_length = float(np.sum(np.linalg.norm(deltas, axis=1)))
    net_displacement = float(np.linalg.norm(root_xy[end] - root_xy[start]))
    clearances = [event.maximum_foot_height_m - 0.055 for event in selected]
    placements = [event.forward_placement_m for event in selected]
    success = (
        len(selected) >= required_steps
        and all(a.side != b.side for a, b in zip(selected, selected[1:]))
        and {event.side for event in selected} == {"left", "right"}
        and min(clearances) >= 0.030
        and min(placements) >= 0.100
        and path_length >= 2.0
    )
    result = BehavioralGaitResult(
        source="LAFAN1 walk1_subject1 retargeted by LocoMuJoCo",
        robot="Unitree H1",
        start_frame=start,
        end_frame=end,
        duration_s=(end - start) / frequency,
        alternating_touchdowns=len(selected),
        completed_strides=len(selected) // 2,
        left_touchdowns=sum(event.side == "left" for event in selected),
        right_touchdowns=sum(event.side == "right" for event in selected),
        minimum_swing_clearance_m=min(clearances),
        minimum_forward_placement_m=min(placements),
        pelvis_path_length_m=path_length,
        net_pelvis_displacement_m=net_displacement,
        success=success,
    )
    return result, selected


def load_lafan_h1():
    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf

    env = ImitationFactory.make(
        "UnitreeH1",
        lafan1_dataset_conf=LAFAN1DatasetConf(["walk1_subject1"]),
        n_substeps=20,
    )
    trajectory = env.th.traj.data
    return env, np.asarray(trajectory.qpos), np.asarray(trajectory.site_xpos), float(env.th.traj.info.frequency)


def render_segment(env, qpos: np.ndarray, start: int, end: int, frequency: float, output: Path) -> None:
    import mujoco

    width, height, fps = 960, 540, 25
    model = env._model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    camera.distance = 4.2
    camera.azimuth = 145
    camera.elevation = -12
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        _ffmpeg_executable(), "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)
    assert process.stdin is not None
    next_video_time = 0.0
    try:
        for frame in range(start, end + 1):
            elapsed = (frame - start) / frequency
            if elapsed + 1e-9 < next_video_time:
                continue
            data.qpos[:] = qpos[frame]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3]
            camera.lookat[2] = 0.85
            renderer.update_scene(data, camera=camera)
            image = renderer.render().copy()
            image[:9, :, :] = (40, 215, 100)
            process.stdin.write(image.tobytes())
            next_video_time += 1.0 / fps
    finally:
        renderer.close()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("연속 보행 영상 인코딩 실패")


def main() -> None:
    parser = argparse.ArgumentParser(description="사람 행동 기준 연속 교대 보행 평가")
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_behavioral_walk.mp4"))
    args = parser.parse_args()
    env, qpos, sites, frequency = load_lafan_h1()
    result, events = evaluate_behavioral_walk(sites, qpos[:, :2], frequency, required_steps=10)
    render_segment(env, qpos, result.start_frame, result.end_frame, frequency, args.output)
    payload = {
        "definition": "좌우 발이 번갈아 이륙하고 진행 방향 앞쪽에 착지하는 동작이 10걸음 이상 연속됨",
        "result": asdict(result),
        "touchdowns": [asdict(event) for event in events],
        "video": str(args.output),
    }
    metrics = args.output.with_suffix(".json")
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preview = args.output.with_suffix(".png")
    subprocess.run([
        _ffmpeg_executable(), "-loglevel", "error", "-y", "-ss", "4.0", "-i",
        str(args.output), "-frames:v", "1", str(preview),
    ], check=True)
    print(json.dumps({**payload, "preview": str(preview)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
