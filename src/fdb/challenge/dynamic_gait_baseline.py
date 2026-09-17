from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import subprocess

import numpy as np

from fdb.challenge.behavioral_gait import (
    detect_touchdowns,
    evaluate_behavioral_walk,
    load_lafan_h1,
    longest_alternating_run,
    render_segment,
)
from fdb.v2.render_final import _ffmpeg_executable


def pd_action(
    position: np.ndarray,
    velocity: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    kp: float,
    kd: float,
    limits: np.ndarray,
) -> np.ndarray:
    value = kp * (target_position - position) + kd * (target_velocity - velocity)
    return np.clip(value, limits[:, 0], limits[:, 1])


def run_baseline(metrics_path: Path) -> dict[str, object]:
    import mujoco

    env, all_qpos, source_sites, frequency = load_lafan_h1()
    reference, _ = evaluate_behavioral_walk(source_sites, all_qpos[:, :2], frequency)
    target = all_qpos[reference.start_frame:reference.end_frame + 1]
    joint_velocity = np.gradient(target[:, 7:], 1.0 / frequency, axis=0)
    candidates = ((120.0, 8.0), (200.0, 12.0), (300.0, 18.0), (450.0, 28.0))
    summaries: list[dict[str, object]] = []
    best_records: list[np.ndarray] = []
    best_sites: list[np.ndarray] = []
    for kp, kd in candidates:
        data = mujoco.MjData(env._model)
        data.qpos[:] = target[0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(env._model, data)
        records: list[np.ndarray] = []
        sites: list[np.ndarray] = []
        total_steps = round((len(target) - 1) / frequency / env._model.opt.timestep)
        fell = False
        for step in range(total_steps + 1):
            phase = min(step * env._model.opt.timestep * frequency, len(target) - 1.000001)
            frame = int(phase)
            blend = phase - frame
            desired_position = (1.0 - blend) * target[frame, 7:] + blend * target[frame + 1, 7:]
            desired_velocity = (1.0 - blend) * joint_velocity[frame] + blend * joint_velocity[frame + 1]
            data.ctrl[:] = pd_action(
                data.qpos[7:], data.qvel[6:], desired_position, desired_velocity,
                kp, kd, env._model.actuator_ctrlrange,
            )
            mujoco.mj_step(env._model, data)
            if step % round(1.0 / (frequency * env._model.opt.timestep)) == 0:
                records.append(data.qpos.copy())
                sites.append(data.site_xpos.copy())
            if data.qpos[2] < 0.35:
                fell = True
                break
        site_array = np.asarray(sites)
        root_array = np.asarray(records)[:, :2]
        events = detect_touchdowns(site_array, root_array, frequency) if len(records) > 2 else []
        alternating = longest_alternating_run(events, round(1.6 * frequency))
        summary = {
            "kp": kp, "kd": kd, "survival_s": step * env._model.opt.timestep,
            "fell": fell, "alternating_touchdowns_before_fall": len(alternating),
            "minimum_root_height_m": float(min(value[2] for value in records)),
        }
        summaries.append(summary)
        if not best_records or summary["survival_s"] > max(item["survival_s"] for item in summaries[:-1]):
            best_records, best_sites = records, sites

    best = max(summaries, key=lambda item: float(item["survival_s"]))
    result = {
        "experiment": "free-root physics tracking of the neural human-gait reference",
        "required_alternating_steps": 10,
        "candidates": summaries,
        "best": best,
        "success": bool(not best["fell"] and best["alternating_touchdowns_before_fall"] >= 10),
        "interpretation": "kinematic imitation alone is insufficient; contact/IMU feedback and balance learning are required",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    replay = all_qpos.copy()
    replay[:len(best_records)] = np.asarray(best_records)
    video = metrics_path.with_suffix(".mp4")
    render_segment(env, replay, 0, len(best_records) - 1, frequency, video)
    preview = video.with_suffix(".png")
    subprocess.run([_ffmpeg_executable(), "-loglevel", "error", "-y", "-ss", "0.8", "-i", str(video),
                    "-frames:v", "1", str(preview)], check=True)
    result.update({"video": str(video), "preview": str(preview)})
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="H1 사람 궤적의 자유 동역학 PD 기준선")
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/fdb_dynamic_gait_baseline.json"))
    args = parser.parse_args()
    print(json.dumps(run_baseline(args.metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

