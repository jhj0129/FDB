from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CartesianWaypoint:
    name: str
    position: tuple[float, float, float]
    carrying_object: bool = False


@dataclass(frozen=True)
class PlanSafetyResult:
    safe: bool
    minimum_hand_clearance_m: float
    minimum_object_clearance_m: float | None
    violations: tuple[str, ...]


def validate_manipulation_plan(
    waypoints: Iterable[CartesianWaypoint],
    *,
    surface_z: float,
    hand_safety_radius_m: float = 0.075,
    object_bottom_offset_m: float = 0.10,
    minimum_clearance_m: float = 0.005,
) -> PlanSafetyResult:
    """Reject Cartesian plans that would bury the hand or a carried object in a surface.

    The test is deliberately geometry based and runs before IK or simulation. Contact during the
    intended final insertion is permitted for the object, but the hand envelope always stays clear.
    """
    points = tuple(waypoints)
    if not points:
        return PlanSafetyResult(False, float("-inf"), None, ("계획에 경유점이 없습니다.",))

    hand_clearances: list[float] = []
    object_clearances: list[float] = []
    violations: list[str] = []
    for point in points:
        hand_clearance = point.position[2] - hand_safety_radius_m - surface_z
        hand_clearances.append(hand_clearance)
        if hand_clearance < minimum_clearance_m:
            violations.append(
                f"{point.name}: 손 안전 외피의 바닥 여유 {hand_clearance:.4f}m가 최소값보다 작습니다."
            )
        if point.carrying_object:
            object_clearance = point.position[2] - object_bottom_offset_m - surface_z
            object_clearances.append(object_clearance)
            # 'insert'만 물체가 수용구 바닥 높이에 닿는 것을 허용한다.
            if point.name != "insert" and object_clearance < minimum_clearance_m:
                violations.append(
                    f"{point.name}: 운반 물체의 바닥 여유 {object_clearance:.4f}m가 최소값보다 작습니다."
                )

    return PlanSafetyResult(
        safe=not violations,
        minimum_hand_clearance_m=min(hand_clearances),
        minimum_object_clearance_m=min(object_clearances) if object_clearances else None,
        violations=tuple(violations),
    )


def require_safe_manipulation_plan(*args, **kwargs) -> PlanSafetyResult:
    result = validate_manipulation_plan(*args, **kwargs)
    if not result.safe:
        raise ValueError("위험한 조작 계획을 실행 전에 거부했습니다: " + " ".join(result.violations))
    return result

