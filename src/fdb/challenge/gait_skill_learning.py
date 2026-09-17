from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
import subprocess

import numpy as np

from fdb.challenge.behavioral_gait import (
    evaluate_behavioral_walk,
    load_lafan_h1,
    render_segment,
)
from fdb.v2.render_final import _ffmpeg_executable


def time_features(frame_count: int, harmonics: int = 10) -> np.ndarray:
    time = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
    values = [time]
    for index in range(1, harmonics + 1):
        values.extend((np.sin(2 * np.pi * index * time), np.cos(2 * np.pi * index * time)))
    return np.column_stack(values).astype(np.float32)


def build_gait_skill(output_size: int):
    from flax import linen as nn

    class GaitSkill(nn.Module):
        @nn.compact
        def __call__(self, features):
            value = nn.tanh(nn.Dense(192)(features))
            value = nn.tanh(nn.Dense(192)(value))
            value = nn.tanh(nn.Dense(128)(value))
            return nn.Dense(output_size)(value)

    return GaitSkill()


def train_gait_skill(model_path: Path, metrics_path: Path, seed: int = 73) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from flax.serialization import to_bytes
    from flax.training import train_state
    import mujoco
    import optax

    env, source_qpos, source_sites, frequency = load_lafan_h1()
    reference, _ = evaluate_behavioral_walk(source_sites, source_qpos[:, :2], frequency)
    target = source_qpos[reference.start_frame:reference.end_frame + 1].astype(np.float32)
    features = time_features(len(target))
    mean = target.mean(axis=0)
    std = target.std(axis=0) + 1e-5
    normalized = (target - mean) / std
    model = build_gait_skill(target.shape[1])
    params = model.init(jax.random.PRNGKey(seed), jnp.asarray(features[:1]))["params"]
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(2e-3, weight_decay=1e-6),
    )

    @jax.jit
    def step(current):
        def loss_fn(parameters):
            prediction = current.apply_fn({"params": parameters}, jnp.asarray(features))
            position_loss = jnp.mean((prediction - jnp.asarray(normalized)) ** 2)
            velocity_loss = jnp.mean((jnp.diff(prediction, axis=0) - jnp.diff(jnp.asarray(normalized), axis=0)) ** 2)
            return position_loss + 0.35 * velocity_loss
        loss, gradients = jax.value_and_grad(loss_fn)(current.params)
        return current.apply_gradients(grads=gradients), loss

    loss_value = math.inf
    for _ in range(4500):
        state, loss = step(state)
        loss_value = float(loss)
    predicted = np.asarray(model.apply({"params": state.params}, jnp.asarray(features))) * std + mean
    quaternion_norm = np.linalg.norm(predicted[:, 3:7], axis=1, keepdims=True)
    predicted[:, 3:7] /= np.maximum(quaternion_norm, 1e-9)

    data = mujoco.MjData(env._model)
    predicted_sites = []
    for qpos in predicted:
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(env._model, data)
        predicted_sites.append(data.site_xpos.copy())
    predicted_sites = np.asarray(predicted_sites)
    generated, events = evaluate_behavioral_walk(predicted_sites, predicted[:, :2], frequency)
    joint_rmse = float(np.sqrt(np.mean((predicted[:, 7:] - target[:, 7:]) ** 2)))
    root_rmse = float(np.sqrt(np.mean((predicted[:, :3] - target[:, :3]) ** 2)))
    metrics = {
        "experiment": "phase-conditioned neural compression of a ten-step human gait skill",
        "source": reference.source,
        "training_frames": len(target),
        "frequency_hz": frequency,
        "parameter_count": int(sum(np.prod(value.shape) for value in jax.tree_util.tree_leaves(state.params))),
        "final_loss": loss_value,
        "joint_pose_rmse_rad": joint_rmse,
        "root_position_rmse_m": root_rmse,
        "generated_behavior": asdict(generated),
        "generated_touchdowns": [asdict(event) for event in events],
        "scope_limit": "사람 궤적을 압축 생성한 kinematic skill이며 물리 외란 대응 정책은 아님",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(to_bytes({"params": state.params, "mean": mean, "std": std}))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 렌더러를 재사용하기 위해 생성 궤적을 원래 배열 크기의 임시 복사본에 놓는다.
    full_prediction = source_qpos.copy()
    full_prediction[reference.start_frame:reference.end_frame + 1] = predicted
    video = metrics_path.with_name("fdb_neural_gait_skill.mp4")
    render_segment(env, full_prediction, reference.start_frame, reference.end_frame, frequency, video)
    preview = video.with_suffix(".png")
    subprocess.run([_ffmpeg_executable(), "-loglevel", "error", "-y", "-ss", "3", "-i", str(video),
                    "-frames:v", "1", str(preview)], check=True)
    metrics["video"] = str(video)
    metrics["preview"] = str(preview)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="사람 10걸음 궤적을 생성하는 신경망 기술 학습")
    parser.add_argument("--model", type=Path, default=Path("models/local/lafan_h1_gait_skill.msgpack"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/fdb_neural_gait_skill.json"))
    args = parser.parse_args()
    print(json.dumps(train_gait_skill(args.model, args.metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
