from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fdb.challenge.behavioral_gait import detect_touchdowns, longest_alternating_run
from fdb.challenge.behavioral_mimic_reward import register_behavioral_mimic_reward


def evaluate_checkpoint(checkpoint: Path, total_steps: int = 3000) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    import mujoco
    from loco_mujoco import TaskFactory
    from loco_mujoco.algorithms import PPOJax
    from omegaconf import OmegaConf

    register_behavioral_mimic_reward()
    agent_conf, agent_state = PPOJax.load_agent(checkpoint)
    config = agent_conf.config
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = True
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    train_state = agent_state.train_state

    @jax.jit
    def deterministic_action(observation):
        output, _ = agent_conf.network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            observation, mutable=["run_stats"],
        )
        return output[0].mode()

    left_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_mimic")
    right_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_mimic")
    episodes: list[tuple[list[np.ndarray], list[np.ndarray]]] = []
    qpos_episode: list[np.ndarray] = []
    sites_episode: list[np.ndarray] = []
    observation = env.reset()
    for _ in range(total_steps):
        action = np.asarray(jnp.atleast_2d(deterministic_action(jnp.asarray(observation))))
        observation, _, _, done, _ = env.step(action)
        qpos_episode.append(env._data.qpos.copy())
        # Preserve the evaluator's canonical left=3/right=6 layout without assuming model site IDs.
        canonical = np.zeros((7, 3), dtype=float)
        canonical[3] = env._data.site_xpos[left_id]
        canonical[6] = env._data.site_xpos[right_id]
        sites_episode.append(canonical)
        if done:
            episodes.append((qpos_episode, sites_episode))
            qpos_episode, sites_episode = [], []
            observation = env.reset()
    if qpos_episode:
        episodes.append((qpos_episode, sites_episode))
    best_qpos, best_sites = max(episodes, key=lambda episode: len(episode[0]))
    qpos = np.asarray(best_qpos)
    sites = np.asarray(best_sites)
    # PDControl performs its own internal physics substeps, so _n_substeps is not
    # the control interval. env.dt is authoritative for both torque and PD modes.
    frequency = 1.0 / env.dt
    touchdowns = detect_touchdowns(sites, qpos[:, :2], frequency)
    alternating = longest_alternating_run(touchdowns, round(1.6 * frequency))
    path = float(np.sum(np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1))) if len(qpos) > 1 else 0.0
    success = bool(len(alternating) >= 10 and len(best_qpos) / frequency >= 5.0 and path >= 2.0)
    return {
        "checkpoint": str(checkpoint),
        "evaluation_steps": total_steps,
        "episodes": len(episodes),
        "best_episode_steps": len(best_qpos),
        "control_frequency_hz": frequency,
        "best_survival_s": len(best_qpos) / frequency,
        "pelvis_path_length_m": path,
        "detected_touchdowns": len(touchdowns),
        "longest_alternating_touchdowns": len(alternating),
        "touchdowns": [asdict(event) for event in alternating],
        "required_alternating_touchdowns": 10,
        "success": success,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepMimic 정책의 엄격한 연속 교대 보행 평가")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_deepmimic_evaluation.json"))
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
