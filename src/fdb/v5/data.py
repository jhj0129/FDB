from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np

from fdb.v1.environment import MujocoPushEnvironment
from fdb.v1.models import PhysicsScenario, PushCandidate


FEATURE_NAMES = (
    "initial_x_m",
    "initial_y_m",
    "target_x_m",
    "target_y_m",
    "mass_scale",
    "sliding_friction",
    "force_newtons",
    "force_steps",
    "settle_steps",
)
TARGET_NAMES = ("final_x_m", "final_y_m")


@dataclass(frozen=True)
class PushDataset:
    features: np.ndarray
    targets: np.ndarray
    success: np.ndarray
    split: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    target_names: tuple[str, ...] = TARGET_NAMES

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            targets=self.targets,
            success=self.success,
            split=self.split,
            feature_names=np.asarray(self.feature_names),
            target_names=np.asarray(self.target_names),
        )

    @classmethod
    def load(cls, path: Path) -> "PushDataset":
        with np.load(path) as payload:
            return cls(
                features=payload["features"],
                targets=payload["targets"],
                success=payload["success"],
                split=payload["split"],
                feature_names=tuple(str(value) for value in payload["feature_names"]),
                target_names=tuple(str(value) for value in payload["target_names"]),
            )


def _outside_interval(generator: random.Random, low: float, high: float, margin: float) -> float:
    if generator.random() < 0.5:
        return generator.uniform(max(0.02, low - margin), low - 0.02)
    return generator.uniform(high + 0.02, high + margin)


def _sample_case(generator: random.Random, *, out_of_distribution: bool) -> tuple[PhysicsScenario, PushCandidate]:
    if out_of_distribution:
        mass = _outside_interval(generator, 0.4, 1.8, 0.25)
        friction = _outside_interval(generator, 0.2, 1.4, 0.3)
        offset_limit = 0.22
    else:
        mass = generator.uniform(0.4, 1.8)
        friction = generator.uniform(0.2, 1.4)
        offset_limit = 0.15
    scenario = PhysicsScenario(
        scenario_id="generated",
        mass_scale=mass,
        sliding_friction=friction,
        initial_offset_x=generator.uniform(-offset_limit, offset_limit),
        initial_offset_y=generator.uniform(-offset_limit, offset_limit),
    )
    plan = PushCandidate(
        plan_id="generated",
        force_newtons=generator.uniform(1.5, 10.0),
        force_steps=generator.randint(50, 300),
        settle_steps=generator.randint(150, 550),
        rationale="무작위 학습 데이터",
    )
    return scenario, plan


def generate_dataset(
    *,
    train_count: int = 2400,
    validation_count: int = 500,
    test_count: int = 500,
    ood_count: int = 500,
    seed: int = 20260917,
    progress_every: int = 250,
) -> PushDataset:
    """MuJoCo에서 독립적인 open-loop push 표본을 생성한다."""
    generator = random.Random(seed)
    environment = MujocoPushEnvironment()
    features: list[list[float]] = []
    targets: list[list[float]] = []
    successes: list[bool] = []
    splits: list[str] = []
    split_counts = (
        ("train", train_count, False),
        ("validation", validation_count, False),
        ("test", test_count, False),
        ("ood", ood_count, True),
    )
    total = sum(count for _, count, _ in split_counts)
    completed = 0
    for split_name, count, is_ood in split_counts:
        for _ in range(count):
            scenario, plan = _sample_case(generator, out_of_distribution=is_ood)
            rollout = environment.simulate(plan, scenario)
            features.append([
                *rollout.initial_position,
                *rollout.target_position,
                scenario.mass_scale,
                scenario.sliding_friction,
                plan.force_newtons,
                float(plan.force_steps),
                float(plan.settle_steps),
            ])
            targets.append(list(rollout.final_position))
            successes.append(rollout.evaluation.success)
            splits.append(split_name)
            completed += 1
            if progress_every and (completed % progress_every == 0 or completed == total):
                print(f"데이터 생성 {completed}/{total} ({100 * completed / total:.1f}%)", flush=True)
    return PushDataset(
        features=np.asarray(features, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        success=np.asarray(successes, dtype=np.bool_),
        split=np.asarray(splits),
    )
