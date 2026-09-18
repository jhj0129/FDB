import pytest

from fdb.challenge.shape_command import execute_shape_command, interpret_shape_command


def test_korean_square_into_square_command_is_understood_and_executed():
    goal, result, _, _ = execute_shape_command("네모를 네모칸에 넣어")
    assert goal.object_shape == "square"
    assert goal.receptacle_shape == "square"
    assert result.success


def test_korean_commands_execute_circle_triangle_and_rectangle():
    for command, expected in (
        ("동그라미를 원형 칸에 넣어", "circle"),
        ("세모를 삼각형 칸에 끼워", "triangle"),
        ("직사각형을 직사각형 구멍에 삽입해", "rectangle"),
    ):
        goal, result, _, _ = execute_shape_command(command)
        assert goal.object_shape == expected
        assert result.perceived_object == expected
        assert result.success


def test_ambiguous_or_incomplete_commands_are_rejected():
    with pytest.raises(ValueError):
        interpret_shape_command("네모를 옮겨")
    with pytest.raises(ValueError):
        interpret_shape_command("네모를 넣어")


def test_command_that_disagrees_with_camera_is_not_silently_executed():
    with pytest.raises(RuntimeError, match="명령 물체"):
        execute_shape_command("세모를 네모칸에 넣어")
