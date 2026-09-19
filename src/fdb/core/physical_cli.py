from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import cv2

from fdb.memory import EpisodeStore

from .atomic_skills import build_shape_manipulation_skills
from .models import Goal
from .physical_runtime import AtomicSkillWorldModel, PhysicalAtomicRuntime, PhysicalSafetyGate
from .physical_scene import PhysicalPersistentShapeEnvironment


def run_tasks(
    *, goals: tuple[str, ...], seed: int, output_directory: Path,
    use_neural: bool = True, use_memory: bool = True,
    position_noise_m: float = 0.0, target_noise_m: float = 0.0,
    inject_failures: bool = False,
    observation_mode: str = "segmentation",
) -> dict[str, object]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    output_directory.mkdir(parents=True, exist_ok=True)
    environment = PhysicalPersistentShapeEnvironment(
        seed=seed, position_noise_m=position_noise_m, target_noise_m=target_noise_m,
        alignment_failure_once={"triangle_01"} if inject_failures else set(),
        grasp_failure_once={"circle_01"} if inject_failures else set(),
        observation_mode=observation_mode,
    )
    skills = build_shape_manipulation_skills()
    brain = PhysicalAtomicRuntime(
        environment, skills, AtomicSkillWorldModel(skills, use_neural=use_neural),
        PhysicalSafetyGate(environment), EpisodeStore(output_directory / "episodes"), use_memory=use_memory,
    )
    results = []
    for shape in goals:
        result = brain.run(Goal(
            "object_to_target", f"{shape}_01", f"{shape}_target",
            {"inside": True, "released": True},
        ))
        results.append({"shape": shape, **asdict(result)})
    rgb = environment.camera_rgb()
    image_path = output_directory / "final_camera.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    payload = {
        "schema_version": "1.0", "mode": "neural+rules" if use_neural else "rules_only",
        "memory": use_memory, "seed": seed, "headless": True,
        "observation_mode": observation_mode,
        "position_noise_m": position_noise_m, "target_noise_m": target_noise_m,
        "failure_injection": inject_failures, "goal_order": list(goals),
        "single_world_reset_count": environment.reset_count,
        "final_scene_sequence": environment.sequence,
        "successes": sum(result["success"] for result in results), "total": len(results),
        "results": results, "final_camera": str(image_path),
        "world_configuration": asdict(environment.configuration),
    }
    (output_directory / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="FDB Phase 2 단일 물리 세계 원자 Skill 실행")
    parser.add_argument("--goals", default="triangle,circle,rectangle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--world-model", choices=("neural", "deterministic"), default="neural")
    parser.add_argument("--memory", choices=("on", "off"), default="on")
    parser.add_argument("--position-noise-m", type=float, default=0.0)
    parser.add_argument("--target-noise-m", type=float, default=0.0)
    parser.add_argument("--inject-failures", action="store_true")
    parser.add_argument("--observation-mode", choices=("oracle", "segmentation", "camera"), default="camera")
    parser.add_argument("--output-directory", type=Path, default=Path("artifacts/phase2_physical"))
    args = parser.parse_args()
    goals = tuple(item.strip() for item in args.goals.split(",") if item.strip())
    invalid = set(goals).difference(PhysicalPersistentShapeEnvironment.SHAPES)
    if invalid:
        parser.error(f"unsupported shapes: {sorted(invalid)}")
    payload = run_tasks(
        goals=goals, seed=args.seed, output_directory=args.output_directory,
        use_neural=args.world_model == "neural", use_memory=args.memory == "on",
        position_noise_m=args.position_noise_m, target_noise_m=args.target_noise_m,
        inject_failures=args.inject_failures,
        observation_mode=args.observation_mode,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
