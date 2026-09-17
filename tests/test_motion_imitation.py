from pathlib import Path

from fdb.challenge.motion_imitation import human_stride, load_amc, simulate


def test_cmu_motion_is_parsed_into_a_complete_human_stride():
    frames = load_amc()
    stride = human_stride()
    assert len(frames) == 469
    assert set(stride) == {
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_arm", "right_arm",
    }
    assert all(len(values) == 143 for values in stride.values())


def test_human_motion_controller_reduces_reference_error_without_falling():
    baseline, _, _ = simulate("procedural_gait_baseline")
    imitation, _, _ = simulate("human_motion_imitation")
    assert not imitation.fallen
    assert imitation.completed_duration_s >= 3.99
    assert imitation.normalized_human_reference_rmse < baseline.normalized_human_reference_rmse
    assert Path("data/human_motion/cmu/69/69_01.amc").exists()
