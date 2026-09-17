import json

import pytest

pytest.importorskip("mujoco")

from fdb.memory import EpisodeStore
from fdb.v1 import PhysicsDecisionLoop
from fdb.v1.environment import MujocoPushEnvironment
from fdb.v1.planner import PushCandidatePlanner
from fdb.v1.robustness import RobustnessExperiment, generate_scenarios


def test_candidates_are_evaluated_from_fresh_initial_state():
    environment = MujocoPushEnvironment()
    rollouts = [
        environment.simulate(candidate)
        for candidate in PushCandidatePlanner().create_candidates()
    ]

    assert len({rollout.initial_position for rollout in rollouts}) == 1
    assert [rollout.plan.plan_id for rollout in rollouts] == [
        "gentle_long",
        "balanced",
        "firm_short",
    ]
    assert sum(rollout.evaluation.success for rollout in rollouts) == 1


def test_loop_selects_successful_plan_and_records_physics_metrics(tmp_path):
    result = PhysicsDecisionLoop(EpisodeStore(tmp_path)).run()

    assert result.selected_prediction.plan.plan_id == "balanced"
    assert result.selected_prediction.evaluation.success is True
    assert result.execution.evaluation.success is True
    assert result.execution.evaluation.final_distance <= 0.14
    assert result.execution.evaluation.unintended_contact_count == 0
    assert result.execution.evaluation.out_of_bounds is False
    assert result.execution.final_position == pytest.approx(
        result.selected_prediction.final_position
    )

    payload = json.loads(result.episode_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "v1"
    assert payload["decision"]["selected_plan_id"] == "balanced"
    assert payload["result"]["objective_evaluation"]["success"] is True


def test_feedback_improves_first_attempt_success_under_variation():
    result = RobustnessExperiment().run(generate_scenarios(count=30, seed=1701))

    assert result.open_loop_summary.success_rate == pytest.approx(0.8)
    assert result.feedback_summary.success_rate == pytest.approx(1.0)
    assert (
        result.feedback_summary.mean_final_distance
        < result.open_loop_summary.mean_final_distance
    )
