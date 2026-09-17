import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mujoco_menagerie")

from fdb.challenge.human_gait import simulate_gait


def test_g1_contact_gait_survives_small_forward_push():
    result, _, _ = simulate_gait(
        "unitree_g1", "pitch_feedback", push_xy_fraction=(0.05, 0.0)
    )
    assert result.upright_complete is True
    assert result.single_support_fraction >= 0.045


def test_t1_upright_is_not_misreported_as_contact_gait():
    result, _, _ = simulate_gait("booster_t1", "pitch_feedback")
    assert result.upright_complete is True
    assert result.success is False


@pytest.mark.parametrize("push", [(0.07, 0.0), (0.0, 0.07)])
def test_op3_momentum_feedback_recovers_from_holdout_pushes(push):
    result, _, _ = simulate_gait(
        "robotis_op3", "momentum_feedback", push_xy_fraction=push
    )
    assert result.upright_complete is True
    assert result.success is True
