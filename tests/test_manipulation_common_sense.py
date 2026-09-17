import pytest

from fdb.challenge.manipulation_common_sense import (
    CartesianWaypoint,
    require_safe_manipulation_plan,
    validate_manipulation_plan,
)


def test_safe_plan_preserves_clearance() -> None:
    result = validate_manipulation_plan(
        (
            CartesianWaypoint("approach", (0.4, 0.1, 0.55)),
            CartesianWaypoint("lift", (0.4, 0.1, 0.58), True),
            CartesianWaypoint("insert", (0.5, -0.1, 0.445), True),
        ),
        surface_z=0.29,
    )
    assert result.safe
    assert result.minimum_hand_clearance_m > 0.07


def test_hand_below_table_envelope_is_rejected_before_execution() -> None:
    with pytest.raises(ValueError, match="실행 전에 거부"):
        require_safe_manipulation_plan(
            (CartesianWaypoint("bad_grasp", (0.4, 0.1, 0.34)),),
            surface_z=0.29,
        )


def test_carried_object_may_only_touch_surface_at_insert() -> None:
    result = validate_manipulation_plan(
        (CartesianWaypoint("transfer", (0.5, 0.0, 0.39), True),),
        surface_z=0.29,
    )
    assert not result.safe
    assert "운반 물체" in result.violations[0]

