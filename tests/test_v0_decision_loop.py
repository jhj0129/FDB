import json

import pytest

from fdb.memory import EpisodeStore
from fdb.v0 import DecisionLoop
from fdb.v0.interpreter import TaskInterpreter
from fdb.v0.models import Position, SceneObject, TargetZone, WorldState


@pytest.fixture
def world() -> WorldState:
    return WorldState(
        width=10,
        height=10,
        objects=(SceneObject("object-1", "red", Position(1, 2)),),
        zones=(TargetZone("zone-1", "blue", Position(7, 8)),),
    )


def test_korean_task_reaches_target_and_records_episode(tmp_path, world):
    result = DecisionLoop(EpisodeStore(tmp_path)).run(
        "빨간 물체를 파란 영역으로 이동", world
    )

    assert result.evaluation.success is True
    assert result.selected_plan.plan_id == "direct"
    assert result.final_state.object_by_color("red").position == Position(7, 8)
    assert result.episode_path is not None
    payload = json.loads(result.episode_path.read_text(encoding="utf-8"))
    assert payload["decision"]["selected_plan_id"] == "direct"
    assert len(payload["candidate_plans"]) == 3


def test_english_task_is_understood(world):
    result = DecisionLoop().run("Move the red object to the blue zone", world)
    assert result.evaluation.success is True


def test_ambiguous_task_fails_explicitly():
    with pytest.raises(ValueError, match="object color"):
        TaskInterpreter().interpret("Move the object")


def test_world_rejects_out_of_bounds_move(world):
    with pytest.raises(ValueError, match="outside world bounds"):
        world.validate_position(Position(99, 99))

