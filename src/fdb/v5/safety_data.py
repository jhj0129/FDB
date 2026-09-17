from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np

from fdb.v2.inspector import load_robot
from fdb.v2.pose_reach import PandaPoseReachExperiment
from fdb.v2.reach import ReachCandidate


SAFETY_FEATURE_NAMES = ("hand_x_m", "hand_y_m", "hand_z_m", "gripper_open")


@dataclass(frozen=True)
class ManipulationSafetyDataset:
    features: np.ndarray
    safe: np.ndarray
    contact_steps: np.ndarray
    deepest_penetration_m: np.ndarray
    split: np.ndarray

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            safe=self.safe,
            contact_steps=self.contact_steps,
            deepest_penetration_m=self.deepest_penetration_m,
            split=self.split,
            feature_names=np.asarray(SAFETY_FEATURE_NAMES),
        )

    @classmethod
    def load(cls, path: Path) -> "ManipulationSafetyDataset":
        with np.load(path) as payload:
            return cls(
                payload["features"], payload["safe"], payload["contact_steps"],
                payload["deepest_penetration_m"], payload["split"],
            )


def generate_safety_dataset(
    *, count: int = 180, seed: int = 20260918, progress_every: int = 20
) -> ManipulationSafetyDataset:
    import mujoco

    generator = random.Random(seed)
    record, _ = load_robot()
    spec = record.spec(record.default_model)
    spec.worldbody.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.5, 0.0, 0.29],
        size=[0.4, 0.4, 0.02], friction=[1.0, 0.005, 0.0001],
    )
    model = spec.compile()
    table_id = model.geom("table").id
    ik_plan = ReachCandidate("안전_데이터_IK", 0.05, 300)
    features: list[list[float]] = []
    safe_labels: list[bool] = []
    contact_counts: list[int] = []
    penetrations: list[float] = []
    splits: list[str] = []
    for index in range(count):
        # 절반은 테이블 근처, 절반은 명백히 여유 있는 높이에서 뽑는다.
        z = generator.uniform(0.34, 0.46) if index % 2 == 0 else generator.uniform(0.46, 0.62)
        x = generator.uniform(0.40, 0.63)
        y = generator.uniform(-0.20, 0.20)
        gripper_open = bool(generator.getrandbits(1))
        target_qpos = PandaPoseReachExperiment((x, y, z)).run_candidate(ik_plan).target_qpos
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        start_arm = data.ctrl[:7].copy()
        start_gripper = float(data.ctrl[7])
        target_gripper = 255.0 if gripper_open else 0.0
        contact_steps = 0
        deepest = 0.0
        for step in range(1000):
            phase = (step + 1) / 1000
            blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
            data.ctrl[:7] = start_arm + blend * (target_qpos - start_arm)
            data.ctrl[7] = start_gripper + blend * (target_gripper - start_gripper)
            mujoco.mj_step(model, data)
            contacted = False
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                if table_id in {int(contact.geom1), int(contact.geom2)}:
                    contacted = True
                    deepest = max(deepest, max(0.0, -float(contact.dist)))
            contact_steps += int(contacted)
        features.append([x, y, z, float(gripper_open)])
        safe_labels.append(contact_steps == 0)
        contact_counts.append(contact_steps)
        penetrations.append(deepest)
        if index < int(count * 0.65):
            splits.append("train")
        elif index < int(count * 0.825):
            splits.append("validation")
        else:
            splits.append("test")
        if progress_every and ((index + 1) % progress_every == 0 or index + 1 == count):
            safe_count = sum(safe_labels)
            print(f"안전 자세 데이터 {index + 1}/{count}, 안전 {safe_count}, 충돌 {index + 1 - safe_count}", flush=True)
    return ManipulationSafetyDataset(
        np.asarray(features, dtype=np.float32),
        np.asarray(safe_labels, dtype=np.bool_),
        np.asarray(contact_counts, dtype=np.int32),
        np.asarray(penetrations, dtype=np.float32),
        np.asarray(splits),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Panda 테이블 충돌 안전 자세 데이터 생성")
    parser.add_argument("--count", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path("data/v5/panda_table_safety.npz"))
    args = parser.parse_args()
    dataset = generate_safety_dataset(count=args.count)
    dataset.save(args.output)
    print(f"저장: {args.output}")


if __name__ == "__main__":
    main()
