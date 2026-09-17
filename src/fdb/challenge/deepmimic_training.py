from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def training_config(timesteps: int, num_envs: int):
    import loco_mujoco
    from omegaconf import OmegaConf

    repository = Path(loco_mujoco.__file__).resolve().parents[1]
    config = OmegaConf.load(repository / "examples/training_examples/jax_rl_mimic/conf.yaml")
    config.experiment.task_factory.params.lafan1_dataset_conf.dataset_name = ["walk1_subject1"]
    config.experiment.env_params.use_mjwarp = False
    config.experiment.env_params.nconmax = 512
    config.experiment.env_params.horizon = 300
    config.experiment.num_envs = num_envs
    config.experiment.num_steps = 64
    config.experiment.total_timesteps = timesteps
    config.experiment.num_minibatches = 4
    config.experiment.update_epochs = 2
    config.experiment.hidden_layers = [128, 128]
    config.experiment.validation.active = False
    config.experiment.validation.num = 1
    config.experiment.validation.num_envs = 1
    config.experiment.validation.num_steps = 10
    config.experiment.debug = True
    return config


def train(timesteps: int, num_envs: int, model_dir: Path, metrics_path: Path) -> dict[str, object]:
    import jax
    from loco_mujoco import TaskFactory
    from loco_mujoco.algorithms import PPOJax

    config = training_config(timesteps, num_envs)
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    agent_conf = PPOJax.init_agent_conf(env, config)
    train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf))
    started = time.monotonic()
    output = train_fn(jax.random.PRNGKey(91))
    jax.block_until_ready(output["agent_state"].train_state.params)
    elapsed = time.monotonic() - started
    model_dir.mkdir(parents=True, exist_ok=True)
    saved = PPOJax.save_agent(model_dir, agent_conf, output["agent_state"])
    returns = np.asarray(output["training_metrics"].mean_episode_return)
    lengths = np.asarray(output["training_metrics"].mean_episode_length)
    result = {
        "algorithm": "LocoMuJoCo PPO DeepMimic",
        "dataset": "LAFAN1 walk1_subject1",
        "timesteps": timesteps,
        "parallel_environments": num_envs,
        "elapsed_s": elapsed,
        "updates": int(config.experiment.num_updates),
        "last_mean_episode_return": float(returns[-1]),
        "last_mean_episode_length": float(lengths[-1]),
        "checkpoint": str(saved),
        "status": "smoke training only; behavioral 10-step evaluation still required",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="H1 사람 보행 DeepMimic PPO 단계 학습")
    parser.add_argument("--timesteps", type=int, default=32768)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--model-dir", type=Path, default=Path("models/local/deepmimic"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/fdb_deepmimic_training.json"))
    args = parser.parse_args()
    print(json.dumps(train(args.timesteps, args.num_envs, args.model_dir, args.metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

