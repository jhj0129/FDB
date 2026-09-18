import numpy as np

from fdb.challenge.behavioral_mimic_reward import contact_agreement


def test_exact_alternating_contact_gets_both_rewards() -> None:
    dense, exact = contact_agreement(True, False, True, False, np)
    assert dense == 1.0
    assert exact == 1.0


def test_double_support_does_not_fake_single_support() -> None:
    dense, exact = contact_agreement(True, True, True, False, np)
    assert dense == 0.5
    assert exact == 0.0


def test_opposite_foot_contact_gets_no_reward() -> None:
    dense, exact = contact_agreement(False, True, True, False, np)
    assert dense == 0.0
    assert exact == 0.0
