import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mujoco_menagerie")

from fdb.v4.motion_probe import MotionProbe


@pytest.mark.parametrize("name", ["franka_emika_panda", "universal_robots_ur5e", "kuka_iiwa_14"])
def test_small_joint_motion_matches_inferred_downstream_tree(name):
    result = MotionProbe().probe_menagerie(name)
    assert result.probe_count >= 6
    assert result.success_rate == 1.0
    assert all(not item.unexpected_affected_bodies for item in result.joint_results)
