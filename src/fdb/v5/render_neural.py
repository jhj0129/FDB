from __future__ import annotations

import argparse
from importlib.resources import as_file, files
import json
import math
from pathlib import Path
import subprocess

from fdb.v1.models import PhysicsScenario
from fdb.v1.planner import PushCandidatePlanner
from fdb.v2.render_final import _ffmpeg_executable
from .hybrid import SafeHybridPushPlanner


def render(output: Path, *, width: int = 640, height: int = 360, fps: int = 30) -> dict[str, object]:
    import mujoco
    import numpy as np

    scenario = PhysicsScenario(scenario_id="신경망_시연", mass_scale=1.25, sliding_friction=0.9, initial_offset_x=0.06, initial_offset_y=-0.04)
    hybrid = SafeHybridPushPlanner(Path("models/v5/push_dynamics_ensemble.npz"))
    decision = hybrid.decide(PushCandidatePlanner().create_candidates(), scenario)
    selected = decision.selected
    if selected.source != "neural_ensemble":
        raise RuntimeError(f"시연 조건에서 신경망이 사용되지 않았습니다: {selected.fallback_reasons}")

    asset = files("fdb.v1").joinpath("assets/push.xml")
    with as_file(asset) as model_path:
        spec = mujoco.MjSpec.from_file(str(model_path))
    spec.worldbody.add_site(
        name="neural_prediction",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[selected.final_position[0], selected.final_position[1], 0.004],
        size=[0.09, 0.003, 0.0],
        rgba=[1.0, 0.75, 0.05, 0.9],
    )
    model = spec.compile()
    puck_id = model.body("puck").id
    puck_geom_id = model.geom("puck_geom").id
    floor_geom_id = model.geom("floor").id
    model.body_mass[puck_id] *= scenario.mass_scale
    model.body_inertia[puck_id, :] *= scenario.mass_scale
    model.geom_friction[puck_geom_id, 0] = scenario.sliding_friction
    model.geom_friction[floor_geom_id, 0] = scenario.sliding_friction
    data = mujoco.MjData(model)
    data.joint("puck_x").qpos[0] = scenario.initial_offset_x
    data.joint("puck_y").qpos[0] = scenario.initial_offset_y
    mujoco.mj_forward(model, data)
    initial = data.body("puck").xpos[:2].copy()
    target = data.site("target").xpos[:2].copy()
    direction = (target - initial) / max(float(np.linalg.norm(target - initial)), 1e-12)

    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            _ffmpeg_executable(), "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-an", "-c:v", "libx264", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("영상 인코더를 열 수 없습니다")
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 0.0]
    camera.distance = 2.45
    camera.azimuth = 90
    camera.elevation = -90
    renderer = mujoco.Renderer(model, height=height, width=width)

    def frame(repeat: int = 1) -> None:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render().tobytes()
        for _ in range(repeat):
            process.stdin.write(pixels)

    try:
        frame(fps)
        next_frame = 0.0
        interval = 1.0 / (fps * 2.0)  # 2배 느리게 표시한다.
        total_steps = selected.plan.force_steps + selected.plan.settle_steps
        for step in range(total_steps):
            data.xfrc_applied[puck_id, :] = 0.0
            if step < selected.plan.force_steps:
                data.xfrc_applied[puck_id, :2] = direction * selected.plan.force_newtons
            mujoco.mj_step(model, data)
            if data.time >= next_frame:
                frame()
                next_frame += interval
        frame(fps * 2)
    finally:
        renderer.close()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("신경망 시연 영상 인코딩 실패")

    actual = data.body("puck").xpos[:2]
    error = math.hypot(float(actual[0]) - selected.final_position[0], float(actual[1]) - selected.final_position[1])
    metrics: dict[str, object] = {
        "model_used": "PushDynamicsMLPEnsemble",
        "selection_source": selected.source,
        "selected_plan": selected.plan.plan_id,
        "initial_position": [float(v) for v in initial],
        "target_position": [float(v) for v in target],
        "neural_predicted_final": list(selected.final_position),
        "actual_final": [float(v) for v in actual],
        "neural_to_physics_error_m": error,
        "uncertainty_m": selected.uncertainty_m,
        "physics_success": decision.verified_execution.evaluation.success,
        "fallback_used": False,
        "visual_legend": {"red": "actual puck", "blue": "goal region", "yellow": "neural predicted final position"},
        "video": str(output),
    }
    if error > 0.03 or not decision.verified_execution.evaluation.success:
        raise RuntimeError(f"신경망 시연 검증 실패: {metrics}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="FDB v5 신경망 예측 대 MuJoCo 시연 렌더링")
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_v5_neural_prediction.mp4"))
    args = parser.parse_args()
    metrics = render(args.output)
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frame_path = args.output.with_suffix(".png")
    subprocess.run(
        [_ffmpeg_executable(), "-loglevel", "error", "-y", "-sseof", "-0.2", "-i", str(args.output), "-frames:v", "1", str(frame_path)],
        check=True,
    )
    print(json.dumps({**metrics, "metrics": str(metrics_path), "final_frame": str(frame_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
