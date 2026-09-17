import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mujoco_menagerie")

from fdb.v2.inspector import inspect_robot
from fdb.v2.reach import PandaReachExperiment
from fdb.v2.reach_suite import WORKSPACE_TARGETS, run_reach_suite


def test_inspector_builds_sourced_panda_self_model():
    model = inspect_robot()
    assert model["robot_id"] == "franka_emika_panda"
    assert model["source_artifacts"][0]["license"] == "Apache-2.0"
    assert len(model["joints"]) == 9
    assert model["end_effector_candidates"][0]["body"] == "hand"
    assert model["grippers"][0]["finger_joints"] == [
        "finger_joint1",
        "finger_joint2",
    ]


def test_reach_candidates_respect_limits_and_execute_with_low_error():
    selected, candidates = PandaReachExperiment().run()
    assert len(candidates) == 3
    assert all(candidate.converged for candidate in candidates)
    assert all(candidate.joint_limit_violations == 0 for candidate in candidates)
    assert selected.execution_error_m < 0.01
    assert selected.contact_count == 0


def test_reach_generalizes_across_declared_workspace_suite():
    result = run_reach_suite()
    assert result.trial_count == len(WORKSPACE_TARGETS) == 12
    assert result.success_rate == 1.0
    assert result.max_execution_error_m < 0.01
    assert result.joint_limit_violations == 0
    assert result.contact_count == 0
