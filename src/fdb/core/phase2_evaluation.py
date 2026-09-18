from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .atomic_skills import AtomicSkill
from .models import Goal


SCRIPTED_SEQUENCE = (
    "reach_object", "grasp_object", "lift_object", "align_object",
    "move_to_target", "insert_object", "release_object",
)


def run_scripted_atomic_baseline(
    environment, skills: tuple[AtomicSkill, ...], goals: tuple[Goal, ...],
) -> dict[str, Any]:
    """Deliberately fixed baseline: never branches or replans after failure."""
    skill_map = {skill.name: skill for skill in skills}
    trials = []
    for goal in goals:
        steps = []
        for name in SCRIPTED_SEQUENCE:
            observation = environment.observe()
            candidate = skill_map[name].propose(observation, goal, None)[0]
            result = environment.execute(candidate)
            steps.append({"skill": name, "result": asdict(result)})
        final = environment.observe().objects[goal.object_id]
        success = final.get("inside_target") == goal.target_id and final.get("released", False)
        trials.append({"object_id": goal.object_id, "success": success, "steps": steps})
    return {
        "baseline": "scripted_atomic_sequence", "sequence": list(SCRIPTED_SEQUENCE),
        "successes": sum(item["success"] for item in trials), "total": len(trials),
        "reset_count": environment.reset_count, "trials": trials,
    }


def summarize_runtime_file(path: Path) -> dict[str, Any]:
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "mode": payload["mode"], "memory": payload["memory"],
        "successes": payload["successes"], "total": payload["total"],
        "actions": sum(item["actions"] for item in payload["results"]),
        "replans": sum(item["replans"] for item in payload["results"]),
        "latency_s": sum(item["metrics"]["total_task_latency_s"] for item in payload["results"]),
        "world_model_calls": sum(item["metrics"]["world_model_calls"] for item in payload["results"]),
        "neural_calls": sum(item["metrics"]["neural_world_model_calls"] for item in payload["results"]),
        "memory_hits": sum(item["metrics"]["memory_retrieval_count"] for item in payload["results"]),
    }


def main() -> None:
    import argparse
    import json
    import os
    from .atomic_skills import build_shape_manipulation_skills
    from .physical_scene import PhysicalPersistentShapeEnvironment

    parser = argparse.ArgumentParser(description="고정 원자 순서 baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inject-failures", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    environment = PhysicalPersistentShapeEnvironment(
        seed=args.seed,
        alignment_failure_once={"triangle_01"} if args.inject_failures else set(),
        grasp_failure_once={"circle_01"} if args.inject_failures else set(),
    )
    goals = tuple(Goal(
        "object_to_target", f"{shape}_01", f"{shape}_target",
        {"inside": True, "released": True},
    ) for shape in ("triangle", "circle", "rectangle"))
    payload = run_scripted_atomic_baseline(environment, build_shape_manipulation_skills(), goals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
