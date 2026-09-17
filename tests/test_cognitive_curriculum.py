from fdb.challenge.cognitive_curriculum import CAPABILITIES, build_profile


def test_profile_does_not_claim_a_human_age() -> None:
    profile = build_profile()
    assert profile["human_age_equivalent"] is None
    assert profile["operational_score"] < profile["operational_target"]


def test_weakest_gate_is_dynamic_locomotion() -> None:
    profile = build_profile()
    assert profile["selected_next_gate"]["capability"] == "동역학 보행"
    assert len(profile["capabilities"]) == len(CAPABILITIES)

