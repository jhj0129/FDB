from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fdb.challenge.contact_gait import evaluate_contact_gait
from fdb.v2.render_final import _ffmpeg_executable


DEFAULT_UPSTREAM = Path("/home/hgui/.cache/fdb/unitree_rl_gym")


def gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ])


def _is_floor_contact(data, floor_geom: int, foot_geoms: set[int]) -> bool:
    return any(
        (contact.geom1 == floor_geom and contact.geom2 in foot_geoms)
        or (contact.geom2 == floor_geom and contact.geom1 in foot_geoms)
        for contact in data.contact
    )


def simulate_official_policy(
    upstream: Path,
    duration_s: float = 15.0,
    command_m_s: tuple[float, float, float] = (0.5, 0.0, 0.0),
    phase_period_s: float = 0.8,
    pushes: tuple[tuple[float, float, float, float], ...] = (),
):
    import mujoco
    import torch
    import yaml

    config_path = upstream / "deploy/deploy_mujoco/configs/h1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    policy_path = upstream / "deploy/pre_train/h1/motion.pt"
    xml_path = upstream / "resources/robots/h1/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = float(config["simulation_dt"])
    decimation = int(config["control_decimation"])
    frequency = 1.0 / model.opt.timestep

    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()
    kps = np.asarray(config["kps"], dtype=np.float32)
    kds = np.asarray(config["kds"], dtype=np.float32)
    default = np.asarray(config["default_angles"], dtype=np.float32)
    command = np.asarray(command_m_s, dtype=np.float32)
    command_scale = np.asarray(config["cmd_scale"], dtype=np.float32)
    action = np.zeros(int(config["num_actions"]), dtype=np.float32)
    target = default.copy()
    observation = np.zeros(int(config["num_obs"]), dtype=np.float32)

    floor_geom = 0
    pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_link")
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_link")
    left_geoms = {index for index in range(model.ngeom) if int(model.geom_bodyid[index]) == left_body}
    right_geoms = {index for index in range(model.ngeom) if int(model.geom_bodyid[index]) == right_body}

    qpos_records: list[np.ndarray] = []
    foot_records: list[np.ndarray] = []
    contact_records: list[np.ndarray] = []
    total_steps = round(duration_s * frequency)
    with torch.inference_mode():
        for counter in range(1, total_steps + 1):
            elapsed = counter * model.opt.timestep
            data.xfrc_applied[pelvis_body] = 0.0
            for start_s, duration, force_x, force_y in pushes:
                if start_s <= elapsed < start_s + duration:
                    data.xfrc_applied[pelvis_body, :2] += [force_x, force_y]
            torque = (target - data.qpos[7:]) * kps - data.qvel[6:] * kds
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            if counter % decimation == 0:
                qj = (data.qpos[7:] - default) * float(config["dof_pos_scale"])
                dqj = data.qvel[6:] * float(config["dof_vel_scale"])
                omega = data.qvel[3:6] * float(config["ang_vel_scale"])
                phase = (counter * model.opt.timestep % phase_period_s) / phase_period_s
                observation[:3] = omega
                observation[3:6] = gravity_orientation(data.qpos[3:7])
                observation[6:9] = command * command_scale
                observation[9:19] = qj
                observation[19:29] = dqj
                observation[29:39] = action
                observation[39:41] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
                action = policy(torch.from_numpy(observation).unsqueeze(0)).numpy().squeeze()
                target = action * float(config["action_scale"]) + default
            qpos_records.append(data.qpos.copy())
            foot_records.append(np.stack([data.xpos[left_body], data.xpos[right_body]]))
            contact_records.append(np.array([
                _is_floor_contact(data, floor_geom, left_geoms),
                _is_floor_contact(data, floor_geom, right_geoms),
            ]))
    return (
        model,
        np.asarray(qpos_records),
        np.asarray(foot_records),
        np.asarray(contact_records),
        frequency,
        policy_path,
    )


def render_walk(model, qpos: np.ndarray, frequency: float, output: Path) -> None:
    import mujoco

    width, height, fps = 960, 540, 30
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    camera.distance = 4.0
    camera.azimuth = 135
    camera.elevation = -12
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        _ffmpeg_executable(), "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)
    assert process.stdin is not None
    next_time = 0.0
    try:
        for frame, pose in enumerate(qpos):
            elapsed = frame / frequency
            if elapsed + 1e-9 < next_time:
                continue
            data.qpos[:] = pose
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            camera.lookat[:] = pose[:3]
            camera.lookat[2] = 0.8
            renderer.update_scene(data, camera=camera)
            image = renderer.render().copy()
            image[:9, :, :] = (37, 200, 110)
            process.stdin.write(image.tobytes())
            next_time += 1.0 / fps
    finally:
        renderer.close()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("H1 보행 영상 인코딩에 실패했습니다.")


def run(
    upstream: Path,
    duration_s: float,
    output: Path,
    command_m_s: tuple[float, float, float] = (0.5, 0.0, 0.0),
    phase_period_s: float = 0.8,
) -> dict[str, object]:
    model, qpos, feet, contacts, frequency, policy_path = simulate_official_policy(
        upstream, duration_s, command_m_s, phase_period_s,
    )
    result, detected, alternating = evaluate_contact_gait(
        feet, contacts, qpos[:, :2], frequency, required_steps=10,
    )
    minimum_pelvis_height = float(np.min(qpos[:, 2]))
    final_pelvis_height = float(qpos[-1, 2])
    remained_upright = bool(minimum_pelvis_height >= 0.75 and final_pelvis_height >= 0.75)
    render_walk(model, qpos, frequency, output)
    preview = output.with_suffix(".png")
    subprocess.run([
        _ffmpeg_executable(), "-loglevel", "error", "-y", "-ss", "5", "-i",
        str(output), "-frames:v", "1", str(preview),
    ], check=True)
    revision = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    payload: dict[str, object] = {
        "source": "unitreerobotics/unitree_rl_gym official pretrained H1 policy",
        "source_url": "https://github.com/unitreerobotics/unitree_rl_gym",
        "source_revision": revision,
        "license": "BSD-3-Clause",
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "definition": "실제 바닥 접촉이 해제된 발이 0.15초 이상 공중에 머물고, 1cm 이상 들린 뒤 8cm 이상 앞에 닿는 좌우 교대 동작 10회",
        "command_m_s": list(command_m_s),
        "phase_period_s": phase_period_s,
        "cadence_steps_per_min": result.alternating_steps / result.duration_s * 60.0,
        "minimum_pelvis_height_m": minimum_pelvis_height,
        "final_pelvis_height_m": final_pelvis_height,
        "remained_upright": remained_upright,
        "overall_success": bool(result.success and remained_upright),
        "physics_frequency_hz": frequency,
        "result": asdict(result),
        "detected_steps": [asdict(step) for step in detected],
        "valid_alternating_steps": [asdict(step) for step in alternating],
        "video": str(output),
        "preview": str(preview),
    }
    metrics = output.with_suffix(".json")
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Unitree 공식 H1 정책의 사람 행동식 접촉 보행 검증")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_unitree_h1_walk.mp4"))
    parser.add_argument("--forward", type=float, default=0.5)
    parser.add_argument("--phase-period", type=float, default=0.8)
    args = parser.parse_args()
    payload = run(
        args.upstream, args.duration, args.output,
        command_m_s=(args.forward, 0.0, 0.0), phase_period_s=args.phase_period,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
