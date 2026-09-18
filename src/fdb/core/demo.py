from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fdb.memory import EpisodeStore

from .models import Goal
from .persistent_scene import PersistentShapeEnvironment
from .runtime import FDBRuntime
from .safety import DeterministicSafetyGate
from .skills import ShapeInsertSkill, SkillDescriptor, SkillRegistry
from .world_model import HybridShapeWorldModel


def run_demo(episode_directory: Path, *, inject_recovery: bool = True) -> dict[str, object]:
    environment = PersistentShapeEnvironment(
        initial_bias_by_shape={"triangle": 40.0} if inject_recovery else {}
    )
    registry = SkillRegistry()
    skill = ShapeInsertSkill()
    registry.register(skill, SkillDescriptor(
        skill.name, skill.capability, "adapter", skill.provenance, True,
    ))
    runtime = FDBRuntime(
        environment, registry, HybridShapeWorldModel(), DeterministicSafetyGate(),
        EpisodeStore(episode_directory),
    )
    results = []
    for shape in ("triangle", "circle", "rectangle"):
        result = runtime.run(Goal(
            "object_to_target", f"{shape}_object", f"{shape}_receptacle",
            {"inside": True, "released": True},
        ))
        results.append({"shape": shape, **asdict(result)})
    final = environment.observe()
    return {
        "scene_id": final.scene_id,
        "scene_sequence": final.sequence,
        "successes": sum(bool(item["success"]) for item in results),
        "total": len(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FDB 지속 장면 폐루프 실행")
    parser.add_argument("--episodes", type=Path, default=Path("episodes"))
    parser.add_argument("--output", type=Path, default=Path("experiments/0040_runtime_e2e.results.json"))
    parser.add_argument("--no-injected-recovery", action="store_true")
    args = parser.parse_args()
    payload = run_demo(args.episodes, inject_recovery=not args.no_injected_recovery)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
