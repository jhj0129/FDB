import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mujoco_menagerie")

from fdb.challenge.human_gait import simulate_gait


def test_op3_completes_human_inspired_gait_without_falling():
    result, _, _ = simulate_gait("robotis_op3", "human_like")
    assert result.completed_duration_s >= 3.99
    assert result.fallen is False
    assert result.displacement_xy_m >= 0.01
    assert result.left_foot_vertical_excursion_m > 0.0
    assert result.arm_swing_amplitude_rad > 0.0


def test_static_arm_ablation_removes_arm_motion_command():
    result, _, _ = simulate_gait("unitree_g1", "static_arms", duration_s=1.0)
    assert result.arm_swing_available is True
    assert result.arm_swing_amplitude_rad == 0.0


def test_pitch_feedback_prevents_t1_open_loop_fall_for_four_seconds():
    open_loop, _, _ = simulate_gait("booster_t1", "human_like")
    stabilized, _, _ = simulate_gait("booster_t1", "pitch_feedback")
    assert open_loop.fallen is True
    assert stabilized.fallen is False
    assert stabilized.completed_duration_s >= 3.99
    assert stabilized.ankle_pitch_feedback_gain == 1.0
