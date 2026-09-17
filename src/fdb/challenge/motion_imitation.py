from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess

import numpy as np

from fdb.challenge.human_gait import (
    GAIT_MAPS,
    PITCH_FEEDBACK_GAINS,
    _apply_gait,
)
from fdb.challenge.humanoid_suite import _reset, _root_body_id
from fdb.v2.render_final import _ffmpeg_executable


DEFAULT_AMC = Path(__file__).resolve().parents[3] / "data/human_motion/cmu/69/69_01.amc"
STRIDE_START = 60
STRIDE_LENGTH = 143


@dataclass(frozen=True)
class ImitationResult:
    controller: str
    amplitude_scale: float
    source: str
    duration_s: float
    completed_duration_s: float
    displacement_xy_m: float
    minimum_height_ratio: float
    minimum_upright_cosine: float
    normalized_human_reference_rmse: float
    fallen: bool


def load_amc(path: Path = DEFAULT_AMC) -> list[dict[str, np.ndarray]]:
    frames: list[dict[str, np.ndarray]] = []
    current: dict[str, np.ndarray] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ":")):
            continue
        if line.isdigit():
            if current:
                frames.append(current)
            current = {}
            continue
        fields = line.split()
        current[fields[0]] = np.asarray([float(value) for value in fields[1:]], dtype=float)
    if current:
        frames.append(current)
    if len(frames) < STRIDE_START + STRIDE_LENGTH:
        raise ValueError(f"보행 주기에 필요한 프레임이 부족합니다: {len(frames)}")
    return frames


def _smooth(values: np.ndarray, width: int = 7) -> np.ndarray:
    padded = np.pad(values, (width // 2, width // 2), mode="wrap")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _centered(values: np.ndarray) -> np.ndarray:
    values = _smooth(values)
    centered = values - np.mean(values)
    return centered / max(float(np.max(np.abs(centered))), 1e-9)


def _positive(values: np.ndarray) -> np.ndarray:
    values = _smooth(values)
    return (values - np.min(values)) / max(float(np.max(values) - np.min(values)), 1e-9)


def human_stride(path: Path = DEFAULT_AMC) -> dict[str, np.ndarray]:
    frames = load_amc(path)[STRIDE_START:STRIDE_START + STRIDE_LENGTH]
    channel = lambda joint, axis=0: np.asarray([frame[joint][axis] for frame in frames])
    return {
        "left_hip": _centered(channel("lfemur")),
        "right_hip": _centered(channel("rfemur")),
        "left_knee": _positive(channel("ltibia")),
        "right_knee": _positive(channel("rtibia")),
        "left_ankle": _centered(channel("lfoot")),
        "right_ankle": _centered(channel("rfoot")),
        "left_arm": _centered(channel("lhumerus")),
        "right_arm": _centered(channel("rhumerus")),
    }


def _sample(values: np.ndarray, phase: float) -> float:
    position = phase * len(values)
    low = int(math.floor(position)) % len(values)
    high = (low + 1) % len(values)
    fraction = position - math.floor(position)
    return float(values[low] * (1.0 - fraction) + values[high] * fraction)


def human_reference_ctrl(
    model, base: np.ndarray, time_s: float, stride: dict[str, np.ndarray], amplitude_scale: float = 1.0
) -> np.ndarray:
    gait = GAIT_MAPS["unitree_g1"]
    target = base.copy()
    ramp = min(1.0, max(0.0, (time_s - 0.5) / 0.8))
    phase = ((time_s - 0.5) * 120.0 / STRIDE_LENGTH) % 1.0
    # 사람 파형은 유지하되 사람과 로봇의 질량/다리 비율 차이를 고려해 첫 파일럿은
    # 관절 진폭을 절반으로 제한한다. 이후 잔차 정책이 안전 범위 안에서 확대한다.
    target[gait.hip_pitch[0]] += amplitude_scale * ramp * 0.020 * _sample(stride["left_hip"], phase)
    target[gait.hip_pitch[1]] += amplitude_scale * ramp * 0.020 * _sample(stride["right_hip"], phase)
    target[gait.knee[0]] += amplitude_scale * ramp * 0.080 * _sample(stride["left_knee"], phase)
    target[gait.knee[1]] += amplitude_scale * ramp * 0.080 * _sample(stride["right_knee"], phase)
    target[gait.ankle_pitch[0]] += amplitude_scale * ramp * 0.035 * _sample(stride["left_ankle"], phase)
    target[gait.ankle_pitch[1]] += amplitude_scale * ramp * 0.035 * _sample(stride["right_ankle"], phase)
    if gait.shoulder_pitch is not None:
        target[gait.shoulder_pitch[0]] += amplitude_scale * ramp * 0.10 * _sample(stride["left_arm"], phase)
        target[gait.shoulder_pitch[1]] += amplitude_scale * ramp * 0.10 * _sample(stride["right_arm"], phase)
    return np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def _actual_positions(model, data, actuator_ids: list[int]) -> np.ndarray:
    positions = []
    for actuator_id in actuator_ids:
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        positions.append(float(data.qpos[int(model.jnt_qposadr[joint_id])]))
    return np.asarray(positions)


def simulate(
    controller: str, path: Path = DEFAULT_AMC, duration_s: float = 4.0,
    frame_callback=None, amplitude_scale: float = 1.0,
):
    import mujoco
    import mujoco_menagerie as menagerie

    if controller not in {"procedural_gait_baseline", "human_motion_imitation"}:
        raise ValueError(controller)
    model = menagerie.get("unitree_g1").model("scene")
    data = _reset(model)
    root = _root_body_id(model)
    gait = GAIT_MAPS["unitree_g1"]
    selected = [*gait.hip_pitch, *gait.knee, *gait.ankle_pitch]
    base = data.ctrl.copy()
    initial = data.xpos[root].copy()
    initial_height = float(initial[2])
    stride = human_stride(path)
    errors: list[float] = []
    minimum_height_ratio = 1.0
    minimum_upright = 1.0
    fallen = False
    completed = 0.0
    for step in range(max(1, int(duration_s / model.opt.timestep))):
        time_s = step * model.opt.timestep
        reference = human_reference_ctrl(model, base, time_s, stride, amplitude_scale)
        if controller == "human_motion_imitation":
            data.ctrl[:] = reference
        else:
            _apply_gait(data, model, gait, base, time_s, "pitch_feedback")
        rotation = data.xmat[root].reshape(3, 3)
        correction = PITCH_FEEDBACK_GAINS["unitree_g1"] * (
            float(rotation[0, 2]) + 0.05 * float(data.qvel[4])
        )
        data.ctrl[gait.ankle_pitch[0]] += correction
        data.ctrl[gait.ankle_pitch[1]] += correction
        data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)
        completed = (step + 1) * model.opt.timestep
        if time_s >= 1.3:
            actual = _actual_positions(model, data, selected)
            ranges = model.actuator_ctrlrange[selected, 1] - model.actuator_ctrlrange[selected, 0]
            errors.append(float(np.mean(((actual - reference[selected]) / np.maximum(ranges, 1e-6)) ** 2)))
        height_ratio = float(data.xpos[root, 2]) / initial_height
        upright = float(data.xmat[root].reshape(3, 3)[2, 2])
        minimum_height_ratio = min(minimum_height_ratio, height_ratio)
        minimum_upright = min(minimum_upright, upright)
        if frame_callback is not None:
            frame_callback(model, data, time_s)
        if height_ratio < 0.55 or upright < 0.5:
            fallen = True
            break
    displacement = data.xpos[root] - initial
    result = ImitationResult(
        controller=controller,
        amplitude_scale=amplitude_scale if controller == "human_motion_imitation" else 1.0,
        source="CMU subject 69 trial 01, frames 61-203",
        duration_s=duration_s,
        completed_duration_s=completed,
        displacement_xy_m=math.hypot(float(displacement[0]), float(displacement[1])),
        minimum_height_ratio=minimum_height_ratio,
        minimum_upright_cosine=minimum_upright,
        normalized_human_reference_rmse=math.sqrt(float(np.mean(errors))) if errors else math.inf,
        fallen=fallen,
    )
    return result, model, data


def render(
    output: Path, path: Path = DEFAULT_AMC, width: int = 960, height: int = 480,
    fps: int = 30, amplitude_scale: float = 1.15,
):
    import mujoco
    import mujoco_menagerie as menagerie

    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        _ffmpeg_executable(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)
    assert process.stdin is not None
    stride = human_stride(path)
    gait = GAIT_MAPS["unitree_g1"]
    states = []
    for controller in ("procedural_gait_baseline", "human_motion_imitation"):
        model = menagerie.get("unitree_g1").model("scene")
        data = _reset(model)
        root = _root_body_id(model)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.0, 0.0, float(data.xpos[root, 2]) * 0.5]
        camera.distance = float(data.xpos[root, 2]) * 3.0
        camera.azimuth = 135
        camera.elevation = -15
        states.append((controller, model, data, root, data.ctrl.copy(), camera,
                       mujoco.Renderer(model, height=height, width=width // 2)))
    next_frame = 0.0
    try:
        for step in range(int(4.0 / 0.002)):
            time_s = step * 0.002
            for controller, model, data, root, base, _, _ in states:
                reference = human_reference_ctrl(
                    model, base, time_s, stride,
                    amplitude_scale if controller == "human_motion_imitation" else 1.0,
                )
                if controller == "human_motion_imitation":
                    data.ctrl[:] = reference
                else:
                    _apply_gait(data, model, gait, base, time_s, "pitch_feedback")
                rotation = data.xmat[root].reshape(3, 3)
                correction = PITCH_FEEDBACK_GAINS["unitree_g1"] * (
                    float(rotation[0, 2]) + 0.05 * float(data.qvel[4])
                )
                data.ctrl[gait.ankle_pitch[0]] += correction
                data.ctrl[gait.ankle_pitch[1]] += correction
                data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
                mujoco.mj_step(model, data)
            if time_s + 1e-9 >= next_frame:
                panels = []
                for index, (_, _, data, _, _, camera, renderer) in enumerate(states):
                    renderer.update_scene(data, camera=camera)
                    panel = renderer.render().copy()
                    color = (255, 155, 40) if index == 0 else (50, 210, 110)
                    panel[:8, :, :] = color
                    panel[-8:, :, :] = color
                    panels.append(panel)
                process.stdin.write(np.hstack(panels).tobytes())
                next_frame += 1.0 / fps
    finally:
        for *_, renderer in states:
            renderer.close()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("모션 모방 비교 영상 인코딩 실패")


def main() -> None:
    parser = argparse.ArgumentParser(description="CMU 사람 보행과 절차식 보행 비교")
    parser.add_argument("--amc", type=Path, default=DEFAULT_AMC)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_motion_imitation.mp4"))
    parser.add_argument("--amplitude-scale", type=float, default=1.15)
    args = parser.parse_args()
    results = [
        simulate(name, args.amc, amplitude_scale=(args.amplitude_scale if name == "human_motion_imitation" else 1.0))[0]
        for name in ("procedural_gait_baseline", "human_motion_imitation")
    ]
    render(args.output, args.amc, amplitude_scale=args.amplitude_scale)
    payload = {
        "experiment": "CMU human motion imitation pilot",
        "robot": "unitree_g1",
        "results": [asdict(result) for result in results],
        "reference_error_improvement_fraction": 1.0 - (
            results[1].normalized_human_reference_rmse / results[0].normalized_human_reference_rmse
        ),
        "video": str(args.output),
        "layout": {"left_orange": "procedural gait baseline", "right_green": "CMU human motion imitation"},
    }
    metrics = args.output.with_suffix(".json")
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preview = args.output.with_suffix(".png")
    subprocess.run([_ffmpeg_executable(), "-loglevel", "error", "-y", "-ss", "2.0", "-i",
                    str(args.output), "-frames:v", "1", str(preview)], check=True)
    print(json.dumps({**payload, "metrics": str(metrics), "preview": str(preview)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
