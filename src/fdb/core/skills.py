from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import CandidateAction, FailureType, Goal, Observation


@dataclass(frozen=True)
class SkillDescriptor:
    name: str
    capability: str
    status: str
    provenance: tuple[str, ...]
    executable: bool


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, object] = {}
        self._descriptors: dict[str, SkillDescriptor] = {}

    def register(self, skill: object, descriptor: SkillDescriptor) -> None:
        if descriptor.name in self._descriptors:
            raise ValueError(f"duplicate skill: {descriptor.name}")
        self._skills[descriptor.name] = skill
        self._descriptors[descriptor.name] = descriptor

    def register_metadata(self, skills_directory: Path) -> None:
        """Index legacy skill records without pretending they are executable adapters."""
        for path in sorted(skills_directory.glob("*.skill.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            name = payload.get("name") or payload.get("skill_id")
            if name in self._descriptors:
                continue
            capability = payload.get("capability") or payload.get("objective") or "legacy"
            evidence = payload.get("evidence", [])
            if isinstance(evidence, dict):
                evidence = list(evidence.values())
            self._descriptors[name] = SkillDescriptor(
                name=name, capability=str(capability), status=payload.get("status", "unknown"),
                provenance=tuple(str(item) for item in evidence), executable=False,
            )

    def available(self, observation: Observation, goal: Goal) -> list[object]:
        return [skill for skill in self._skills.values() if skill.applicable(observation, goal)]

    def get(self, name: str) -> object:
        return self._skills[name]

    def descriptors(self) -> tuple[SkillDescriptor, ...]:
        return tuple(self._descriptors[name] for name in sorted(self._descriptors))


class ShapeInsertSkill:
    name = "shape_insert"
    capability = "persistent_shape_insertion"
    provenance = (
        "src/fdb/challenge/shape_insertion.py",
        "experiments/0037_multi_shape_physical_insertion.md",
    )

    def applicable(self, observation: Observation, goal: Goal) -> bool:
        return (
            goal.task == "object_to_target"
            and goal.object_id in observation.objects
            and goal.target_id in observation.targets
            and not observation.objects[goal.object_id].get("completed", False)
        )

    def propose(
        self, observation: Observation, goal: Goal, failure: FailureType | None,
    ) -> list[CandidateAction]:
        base = {
            "object_id": goal.object_id,
            "target_id": goal.target_id,
            "clearance_m": 0.10,
        }
        if failure == FailureType.ALIGNMENT_FAILURE:
            biases = (0.0, -10.0, 10.0, -20.0, 20.0)
        else:
            biases = (float(observation.objects[goal.object_id].get("initial_bias_deg", 0.0)),)
        return [
            CandidateAction(
                action_id=f"insert:{goal.object_id}:{bias:+.1f}", skill_name=self.name,
                parameters={**base, "rotation_bias_deg": bias}, confidence=0.95,
            )
            for bias in biases
        ]
