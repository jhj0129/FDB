import json
from pathlib import Path

from jsonschema import validate

from fdb.core.models import CandidateAction, Goal
from fdb.core.persistent_scene import PersistentShapeEnvironment
from fdb.core.runtime import FDBRuntime
from fdb.core.safety import DeterministicSafetyGate
from fdb.core.skills import ShapeInsertSkill, SkillDescriptor, SkillRegistry
from fdb.core.world_model import HybridShapeWorldModel
from fdb.memory import EpisodeStore


def make_runtime(tmp_path: Path, environment: PersistentShapeEnvironment):
    registry = SkillRegistry()
    skill = ShapeInsertSkill()
    registry.register(
        skill,
        SkillDescriptor(skill.name, skill.capability, "adapter", skill.provenance, True),
    )
    return FDBRuntime(
        environment, registry, HybridShapeWorldModel(), DeterministicSafetyGate(),
        EpisodeStore(tmp_path / "episodes"),
    ), registry


def goal(shape: str) -> Goal:
    return Goal(
        "object_to_target", f"{shape}_object", f"{shape}_receptacle",
        {"inside": True, "released": True},
    )


def test_runtime_replans_and_writes_schema_valid_episodes(tmp_path) -> None:
    environment = PersistentShapeEnvironment(initial_bias_by_shape={"triangle": 40.0})
    runtime, _ = make_runtime(tmp_path, environment)
    result = runtime.run(goal("triangle"))
    assert result.success
    assert result.actions == 2
    assert result.replans == 1
    assert result.final_observation.objects["triangle_object"]["completed"]
    schema = json.loads(Path("schemas/episode.schema.json").read_text(encoding="utf-8"))
    for episode_path in result.episode_paths:
        validate(json.loads(Path(episode_path).read_text(encoding="utf-8")), schema)


def test_scene_persists_across_sequential_goals(tmp_path) -> None:
    environment = PersistentShapeEnvironment()
    runtime, _ = make_runtime(tmp_path, environment)
    first = runtime.run(goal("triangle"))
    second = runtime.run(goal("circle"))
    assert first.success and second.success
    assert first.final_observation.scene_id == second.final_observation.scene_id
    assert second.final_observation.sequence == 2
    assert second.final_observation.objects["triangle_object"]["completed"]
    assert second.final_observation.objects["circle_object"]["completed"]


def test_deterministic_gate_rejects_shape_mismatch(tmp_path) -> None:
    environment = PersistentShapeEnvironment()
    runtime, registry = make_runtime(tmp_path, environment)
    skill = registry.get("shape_insert")
    mismatched = Goal(
        "object_to_target", "triangle_object", "circle_receptacle",
        {"inside": True, "released": True},
    )
    candidate = skill.propose(environment.observe(), mismatched, None)[0]
    prediction = runtime.world_model.predict(environment.observe(), candidate)
    decision = runtime.safety_gate.check(environment.observe(), prediction)
    assert not decision.allowed
    assert "SHAPE_MISMATCH" in decision.reasons


def test_goal_does_not_create_or_filter_the_observation(tmp_path) -> None:
    environment = PersistentShapeEnvironment()
    before = environment.observe()
    runtime, _ = make_runtime(tmp_path, environment)
    runtime.run(goal("rectangle"))
    assert set(before.objects) == {
        "square_object", "circle_object", "triangle_object", "rectangle_object"
    }
    assert len(before.targets) == 4


def test_impossible_goal_still_creates_a_failure_episode(tmp_path) -> None:
    environment = PersistentShapeEnvironment()
    runtime, _ = make_runtime(tmp_path, environment)
    result = runtime.run(Goal(
        "object_to_target", "hexagon_object", "hexagon_receptacle",
        {"inside": True, "released": True},
    ))
    assert not result.success
    assert result.metrics["final_failure_type"] == "TARGET_NOT_FOUND"
    assert len(result.episode_paths) == 1
