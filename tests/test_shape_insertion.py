import math

from fdb.challenge.shape_insertion import run_insertion


def test_square_peg_is_aligned_and_inserted_without_robot_table_contact():
    result, _, _ = run_insertion()
    assert result.success
    assert result.final_xy_error_m <= 0.009
    assert result.final_yaw_error_deg <= 8.0
    assert result.robot_table_contact_steps == 0
    assert result.released


def test_camera_neural_policy_generalizes_across_unseen_square_rotations():
    cases = ((-31, 17), (8, -28), (39, 2))
    for object_degrees, hole_degrees in cases:
        result, _, _ = run_insertion(
            object_yaw=math.radians(object_degrees),
            hole_yaw=math.radians(hole_degrees),
        )
        assert result.success, (object_degrees, hole_degrees, result)


def test_camera_neural_policy_executes_all_four_known_physical_shapes():
    for shape in ("square", "circle", "triangle", "rectangle"):
        result, _, _ = run_insertion(shape=shape)
        assert result.perceived_object == shape
        assert result.perceived_hole == shape
        assert result.success, (shape, result)
