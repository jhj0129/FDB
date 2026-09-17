import math

from fdb.challenge.insertion_recovery import recover_from_rotation_error


def test_large_visual_rotation_error_is_detected_and_recovered():
    recovery = recover_from_rotation_error(math.radians(40))
    assert not recovery.initial_success
    assert recovery.recovered_success
    assert recovery.selected_bias_deg is not None
    assert recovery.final_result is not None and recovery.final_result.success
    assert len(recovery.attempted_biases_deg) >= 2
