import pytest

from fdb.challenge.shape_sequence import execute_shape_sequence, interpret_shape_sequence


def test_sequence_is_parsed_in_spoken_order_before_execution():
    clauses = interpret_shape_sequence(
        "먼저 세모를 삼각형 칸에 넣고 다음으로 원을 원형 칸에 넣어"
    )
    assert clauses == ("세모를 삼각형 칸에 넣고", "원을 원형 칸에 넣어")


def test_ambiguous_later_clause_rejects_whole_plan_before_execution():
    with pytest.raises(ValueError, match="물체 형상과 구멍 형상"):
        interpret_shape_sequence("먼저 네모를 네모칸에 넣고 다음으로 원을 넣어")


def test_two_stage_korean_plan_executes_in_order():
    result = execute_shape_sequence(
        "먼저 세모를 삼각형 칸에 넣고 다음으로 원을 원형 칸에 넣어"
    )
    assert [goal.object_shape for goal in result.goals] == ["triangle", "circle"]
    assert result.completed == 2
    assert result.success
