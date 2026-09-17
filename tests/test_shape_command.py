import pytest

from fdb.challenge.shape_command import execute_shape_command, interpret_shape_command


def test_korean_square_into_square_command_is_understood_and_executed():
    goal, result, _, _ = execute_shape_command("네모를 네모칸에 넣어")
    assert goal.object_shape == "square"
    assert goal.receptacle_shape == "square"
    assert result.success


def test_ambiguous_or_incomplete_commands_are_rejected():
    with pytest.raises(ValueError):
        interpret_shape_command("네모를 옮겨")
    with pytest.raises(ValueError):
        interpret_shape_command("네모를 넣어")


def test_command_that_disagrees_with_camera_is_not_silently_executed():
    with pytest.raises(RuntimeError, match="명령 물체"):
        execute_shape_command("세모를 네모칸에 넣어")
