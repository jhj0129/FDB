import json
import math
from pathlib import Path

import pytest
from jsonschema import validate

from fdb.core.atomic_skills import build_shape_manipulation_skills
from fdb.core.models import Goal
from fdb.core.physical_runtime import AtomicSkillWorldModel, PhysicalAtomicRuntime, PhysicalSafetyGate
from fdb.core.physical_scene import PhysicalPersistentShapeEnvironment
from fdb.memory import EpisodeStore


pytestmark = pytest.mark.integration


def goal(shape: str):
    return Goal("object_to_target", f"{shape}_01", f"{shape}_target", {"inside": True, "released": True})


def runtime(tmp_path: Path, environment: PhysicalPersistentShapeEnvironment):
    skills = build_shape_manipulation_skills()
    return PhysicalAtomicRuntime(
        environment, skills, AtomicSkillWorldModel(skills), PhysicalSafetyGate(environment),
        EpisodeStore(tmp_path / "episodes"),
    )


def test_one_physical_world_contains_every_entity_before_goal() -> None:
    environment = PhysicalPersistentShapeEnvironment(seed=7)
    observation = environment.observe()
    assert set(observation.objects) == {"square_01", "circle_01", "triangle_01", "rectangle_01"}
    assert set(observation.targets) == {
        "square_target", "circle_target", "triangle_target", "rectangle_target"
    }
    assert environment.reset_count == 1
    assert all(item["observation_source"].startswith("camera_rgb") for item in observation.objects.values())


def test_camera_only_observation_uses_rgbd_tracks_with_bounded_pose_error() -> None:
    environment = PhysicalPersistentShapeEnvironment(seed=0, observation_mode="camera")
    observation = environment.observe()
    assert len(observation.objects) == len(observation.targets) == 4
    assert all(
        item["observation_source"] == "camera_rgbd_color_contour_tracking"
        for item in (*observation.objects.values(), *observation.targets.values())
    )
    for entity_id, item in {**observation.objects, **observation.targets}.items():
        truth = (
            environment.configuration.object_poses.get(entity_id)
            or environment.configuration.target_poses[entity_id]
        )
        assert math.dist(item["pose"][:2], truth[:2]) < 0.003
        assert item["tracking_age"] == 1


def test_multi_goal_world_recovers_without_reset(tmp_path) -> None:
    environment = PhysicalPersistentShapeEnvironment(
        seed=0, alignment_failure_once={"triangle_01"}, grasp_failure_once={"circle_01"},
    )
    brain = runtime(tmp_path, environment)
    results = [brain.run(goal(shape)) for shape in ("triangle", "circle", "rectangle")]
    assert all(result.success for result in results)
    assert [result.replans for result in results] == [1, 1, 0]
    assert environment.reset_count == 1
    assert environment.sequence == 24
    schema = json.loads(Path("schemas/episode.schema.json").read_text(encoding="utf-8"))
    for result in results:
        episode = json.loads(Path(result.episode_paths[0]).read_text(encoding="utf-8"))
        validate(episode, schema)
        assert episode["decision_steps"]
        assert all("rejected_skills" in step for step in episode["decision_steps"])
