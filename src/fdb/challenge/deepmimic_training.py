from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fdb.challenge.behavioral_mimic_reward import register_behavioral_mimic_reward


def training_config(timesteps: int, num_envs: int, behavioral_reward: bool = False,
                    position_control: bool = False, dataset: str = "walk1_subject1"):
    import loco_mujoco
    from omegaconf import OmegaConf

    repository = Path(loco_mujoco.__file__).resolve().parents[1]
    config = OmegaConf.load(repository / "examples/training_examples/jax_rl_mimic/conf.yaml")
    config.experiment.task_factory.params.lafan1_dataset_conf.dataset_name = [dataset]
    config.experiment.env_params.use_mjwarp = False
    config.experiment.env_params.nconmax = 512
    # Ten alternating human steps take about 5.8 s at 100 Hz control. A 300-step
    # horizon made the requested behavior mathematically impossible to complete.
    config.experiment.env_params.horizon = 800
    if position_control:
        config.experiment.env_params.control_type = config.control_config.position_control.control_type
        config.experiment.env_params.control_params = config.control_config.position_control.control_params
    if behavioral_reward:
        register_behavioral_mimic_reward()
        config.experiment.env_params.reward_type = "BehavioralMimicReward"
        config.experiment.env_params.reward_params.contact_match_weight = 0.45
        config.experiment.env_params.reward_params.single_support_weight = 0.35
        config.experiment.env_params.reward_params.foot_height_weight = 0.25
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


def train(timesteps: int, num_envs: int, model_dir: Path, metrics_path: Path,
          behavioral_reward: bool = False, resume: Path | None = None,
          position_control: bool = False, dataset: str = "walk1_subject1") -> dict[str, object]:
    import jax
    from loco_mujoco import TaskFactory
    from loco_mujoco.algorithms import PPOJax

    config = training_config(timesteps, num_envs, behavioral_reward, position_control, dataset)
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
    if resume is None:
        agent_conf = PPOJax.init_agent_conf(env, config)
        train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf))
        initial_state = None
    else:
        loaded_conf, initial_state = PPOJax.load_agent(resume)
        # Network and optimizer shape stay unchanged; rollout/update counts and the environment use the new stage.
        agent_conf = type(loaded_conf)(config=config, network=loaded_conf.network, tx=loaded_conf.tx)
        PPOJax.init_agent_conf(env, config)  # populate derived counts used by the resumed trainer
        train_fn = jax.jit(PPOJax.build_resume_train_fn(env, agent_conf))
    started = time.monotonic()
    output = (train_fn(jax.random.PRNGKey(91)) if initial_state is None
              else train_fn(jax.random.PRNGKey(91), initial_state))
    jax.block_until_ready(output["agent_state"].train_state.params)
    elapsed = time.monotonic() - started
    model_dir.mkdir(parents=True, exist_ok=True)
    saved = PPOJax.save_agent(model_dir, agent_conf, output["agent_state"])
    returns = np.asarray(output["training_metrics"].mean_episode_return)
    lengths = np.asarray(output["training_metrics"].mean_episode_length)
    result = {
        "algorithm": "LocoMuJoCo PPO DeepMimic",
        "dataset": f"LAFAN1 {dataset}",
        "timesteps": timesteps,
        "parallel_environments": num_envs,
        "elapsed_s": elapsed,
        "updates": int(config.experiment.num_updates),
        "last_mean_episode_return": float(returns[-1]),
        "last_mean_episode_length": float(lengths[-1]),
        "checkpoint": str(saved),
        "resumed_from": str(resume) if resume is not None else None,
        "behavioral_contact_reward": behavioral_reward,
        "control": "PD position" if position_control else "direct torque",
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
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--behavioral-reward", action="store_true")
    parser.add_argument("--position-control", action="store_true")
    parser.add_argument("--dataset", default="walk1_subject1")
    args = parser.parse_args()
    print(json.dumps(train(args.timesteps, args.num_envs, args.model_dir, args.metrics,
                           args.behavioral_reward, args.resume, args.position_control,
                           args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
