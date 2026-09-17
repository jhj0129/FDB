import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mujoco_menagerie")

from fdb.challenge.humanoid_suite import evaluate_humanoid


def test_op3_runs_multiple_applicable_tasks_with_source_provenance():
    result = evaluate_humanoid("robotis_op3")
    applicable = [task for task in result.tasks if task.applicable]
    assert len(applicable) >= 4
    assert {task.task for task in applicable} >= {
        "stand_2s", "scaled_push_recovery", "left_arm_raise", "crouch_candidates"
    }
    assert result.source["repository"].endswith("mujoco_menagerie")
    assert result.source["object_id"]
    assert result.morphology["actuator_count"] == 20


def test_lower_body_model_marks_arm_and_head_tasks_not_applicable():
    result = evaluate_humanoid("berkeley_humanoid")
    task_map = {task.task: task for task in result.tasks}
    assert task_map["left_arm_raise"].applicable is False
    assert task_map["head_turn"].applicable is False
    assert task_map["stand_2s"].applicable is True
    assert task_map["crouch_candidates"].applicable is True
