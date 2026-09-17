from pathlib import Path

from fdb.v5.safety_advisor import PortableSafetyEnsemble


MODEL = Path("models/v5/panda_table_safety.npz")


def test_learned_advisor_rejects_old_penetrating_grasp_and_place():
    advisor = PortableSafetyEnsemble(MODEL)
    old_grasp = advisor.advise([0.48, 0.0, 0.375, 0.0])
    old_place = advisor.advise([0.58, 0.16, 0.385, 1.0])
    assert old_grasp.learned_gate_allows is False
    assert old_place.learned_gate_allows is False
    assert old_grasp.requires_physics_check is False
    assert old_place.requires_physics_check is False


def test_corrected_poses_pass_advisor_but_still_require_physics_check():
    advisor = PortableSafetyEnsemble(MODEL)
    safe_grasp = advisor.advise([0.48, 0.0, 0.435, 0.0])
    safe_place = advisor.advise([0.58, 0.16, 0.465, 1.0])
    assert safe_grasp.learned_gate_allows is True
    assert safe_place.learned_gate_allows is True
    assert safe_grasp.requires_physics_check is True
    assert safe_place.requires_physics_check is True


def test_out_of_distribution_pose_is_rejected_early():
    advisor = PortableSafetyEnsemble(MODEL)
    advice = advisor.advise([0.80, 0.0, 0.38, 0.0])
    assert advice.in_distribution is False
    assert advice.learned_gate_allows is False
    assert "학습 분포 밖 자세" in advice.reasons
