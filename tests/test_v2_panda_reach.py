import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mujoco_menagerie")

from fdb.v2.inspector import inspect_robot
from fdb.v2.reach import PandaReachExperiment
from fdb.v2.reach_suite import WORKSPACE_TARGETS, run_reach_suite
from fdb.v2.pose_reach import PandaPoseReachExperiment
from fdb.v2.pregrasp import CollisionAwarePregraspExperiment
from fdb.v2.grasp_lift import PandaGraspLiftExperiment
from fdb.v2.pick_place import PandaPickPlaceExperiment
from fdb.v2.manipulation_suite import SCENARIOS, run_manipulation_suite
from fdb.v2.recovery import ResetBasedRecoveryExperiment


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


def test_pose_reach_preserves_home_hand_orientation():
    selected, candidates = PandaPoseReachExperiment().run()
    assert any(candidate.converged for candidate in candidates)
    assert selected.position_error_m <= 0.02
    assert selected.orientation_error_rad <= 0.03
    assert selected.joint_limit_violations == 0


def test_grasp_candidates_select_a_retained_lift():
    selected, candidates = PandaGraspLiftExperiment().run()
    assert any(candidate.success for candidate in candidates)
    assert selected.plan.plan_id == "grasp_low"
    assert selected.lift_height_m >= 0.08
    assert selected.final_object_to_hand_distance_m <= 0.16
    assert selected.forbidden_contact_steps == 0


def test_pick_place_selects_a_stable_released_placement():
    selected, candidates = PandaPickPlaceExperiment().run()
    assert any(candidate.success for candidate in candidates)
    assert selected.success is True
    assert selected.pick_lift_height_m >= 0.08
    assert selected.final_xy_error_m <= 0.05
    assert selected.released is True
    assert selected.robot_table_contact_steps == 0
    assert selected.deepest_robot_table_penetration_m == 0.0
    assert selected.stage_sequence_valid is True


def test_pick_place_generalizes_across_declared_object_suite():
    result = run_manipulation_suite()
    assert result.trial_count == len(SCENARIOS) == 6
    assert result.success_rate >= 5 / 6
    assert result.max_final_xy_error_m <= 0.05
    assert all(trial.selected.robot_table_contact_steps == 0 for trial in result.trials)
    assert all(trial.selected.stage_sequence_valid for trial in result.trials)


def test_missed_grasp_is_diagnosed_and_recovers_from_reset():
    result = ResetBasedRecoveryExperiment().run()
    assert result.initial_attempt.success is False
    assert result.diagnosis.failure_type == "missed_grasp"
    assert result.recovery_attempt.success is True
    assert result.recovered is True
    assert result.reset_based is True


def test_pregrasp_rejects_colliding_candidates_before_score():
    selected, candidates = CollisionAwarePregraspExperiment().run()
    by_id = {candidate.plan.plan_id: candidate for candidate in candidates}
    assert by_id["direct_center"].contact_step_count > 0
    assert by_id["centered_high"].contact_step_count > 0
    assert selected.plan.plan_id == "side_high"
    assert selected.collision_free is True
    assert selected.position_error_m <= 0.02
