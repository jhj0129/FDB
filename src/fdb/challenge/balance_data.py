from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np

from .humanoid_suite import DEFAULT_ROBOTS, _actuator_match, _reset, _root_body_id, _simulate


FEATURE_NAMES = (
    "mass_kg", "root_height_m", "actuator_count", "body_count",
    "pre_height_ratio", "pre_upright_cosine", "push_weight_ratio",
    "push_direction_x", "push_direction_y", "push_duration_s",
    "pre_motion_delta_rad", "has_knee_motion",
)


def generate_balance_dataset(per_robot: int = 48, seed: int = 20260917) -> dict[str, np.ndarray]:
    import mujoco
    import mujoco_menagerie as menagerie

    rng = random.Random(seed)
    features, stable, robots, splits = [], [], [], []
    final_height_ratio, final_upright = [], []
    for robot_index, robot_name in enumerate(DEFAULT_ROBOTS):
        model = menagerie.get(robot_name).model("scene")
        root_id = _root_body_id(model)
        mass = float(np.sum(model.body_mass))
        knees = (
            _actuator_match(model, ("left", "knee"))
            + _actuator_match(model, ("right", "knee"))
            + _actuator_match(model, ("l_knee",))
            + _actuator_match(model, ("r_knee",))
            + _actuator_match(model, ("ll_kfe",))
            + _actuator_match(model, ("lr_kfe",))
        )
        knees = list(dict.fromkeys(knees))
        for episode in range(per_robot):
            data = _reset(model)
            initial_height = float(data.xpos[root_id, 2])
            _simulate(model, data, 0.35)
            use_knees = bool(knees and rng.random() < 0.55)
            delta = rng.uniform(-0.38, 0.38) if use_knees else 0.0
            starts = {index: float(data.ctrl[index]) for index in knees} if use_knees else {}
            targets = {
                index: float(np.clip(
                    start + delta,
                    model.actuator_ctrlrange[index, 0],
                    model.actuator_ctrlrange[index, 1],
                ))
                for index, start in starts.items()
            }

            def premotion(step: int, state: object) -> None:
                phase = min(1.0, (step + 1) * model.opt.timestep / 0.45)
                blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
                for index, start in starts.items():
                    state.ctrl[index] = start + blend * (targets[index] - start)

            _simulate(model, data, 0.5, callback=premotion)
            pre_height = float(data.xpos[root_id, 2]) / max(initial_height, 1e-6)
            pre_upright = float(data.xmat[root_id].reshape(3, 3)[2, 2])
            ratio = rng.uniform(0.0, 0.45)
            angle = rng.uniform(-np.pi, np.pi)
            duration = rng.uniform(0.08, 0.23)
            force = ratio * mass * 9.81
            push_steps = int(duration / model.opt.timestep)

            def push(step: int, state: object) -> None:
                state.xfrc_applied[root_id] = 0.0
                if step < push_steps:
                    state.xfrc_applied[root_id, 0] = force * np.cos(angle)
                    state.xfrc_applied[root_id, 1] = force * np.sin(angle)

            # 다음 행동 후보를 고르기 위한 1초 단기 낙상 예측 창이다.
            _simulate(model, data, 1.0, callback=push)
            height_ratio = float(data.xpos[root_id, 2]) / max(initial_height, 1e-6)
            upright = float(data.xmat[root_id].reshape(3, 3)[2, 2])
            is_stable = bool(height_ratio >= 0.55 and upright >= 0.5)
            features.append([
                mass, initial_height, float(model.nu), float(model.nbody - 1),
                pre_height, pre_upright, ratio, float(np.cos(angle)),
                float(np.sin(angle)), duration, delta, float(use_knees),
            ])
            stable.append(is_stable)
            robots.append(robot_name)
            final_height_ratio.append(height_ratio)
            final_upright.append(upright)
            split_roll = episode % 10
            splits.append("test" if split_roll >= 8 else ("validation" if split_roll == 7 else "train"))
            if (episode + 1) % 12 == 0:
                local = stable[-(episode + 1):]
                print(
                    f"균형 데이터 {robot_index + 1}/4 {robot_name} {episode + 1}/{per_robot}, "
                    f"안정 {sum(local)} 낙상 {len(local) - sum(local)}",
                    flush=True,
                )
    return {
        "features": np.asarray(features, dtype=np.float32),
        "stable": np.asarray(stable, dtype=np.bool_),
        "robot": np.asarray(robots),
        "split": np.asarray(splits),
        "final_height_ratio": np.asarray(final_height_ratio, dtype=np.float32),
        "final_upright_cosine": np.asarray(final_upright, dtype=np.float32),
        "feature_names": np.asarray(FEATURE_NAMES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="다중 휴머노이드 균형·낙상 데이터 생성")
    parser.add_argument("--per-robot", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("data/v5/humanoid_balance.npz"))
    args = parser.parse_args()
    payload = generate_balance_dataset(args.per_robot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(
        f"저장: {args.output}, 총 {len(payload['stable'])}, "
        f"안정 {int(payload['stable'].sum())}, 낙상 {int((~payload['stable']).sum())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
