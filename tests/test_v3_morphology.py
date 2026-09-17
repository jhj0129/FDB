import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mujoco_menagerie")

from fdb.v3.morphology import GenericMorphologyInspector


@pytest.mark.parametrize(
    ("name", "joint_count", "license_name"),
    [
        ("franka_emika_panda", 9, "Apache-2.0"),
        ("universal_robots_ur5e", 6, "BSD-3-Clause"),
        ("kuka_iiwa_14", 7, "BSD-3-Clause"),
    ],
)
def test_generic_inspector_reads_multiple_robot_morphologies(name, joint_count, license_name):
    result = GenericMorphologyInspector().inspect_menagerie(name)
    assert len(result["joints"]) == joint_count
    assert result["source_artifacts"][0]["license"] == license_name
    assert result["end_effector_candidates"]
    assert result["confirmed_facts"][0]["actuated_hinge_count"] >= 6
    assert all(joint["source"] == "compiled Menagerie MJCF" for joint in result["joints"])
