from __future__ import annotations

from .models import CandidatePlan, Goal, MoveAction, Position, WorldState


class CandidatePlanner:
    """Creates transparent candidate futures for the v0 symbolic world."""

    def create_candidates(self, world: WorldState, goal: Goal) -> tuple[CandidatePlan, ...]:
        item = world.object_by_color(goal.object_color)
        target = world.zone_by_color(goal.target_color)
        direct = CandidatePlan(
            plan_id="direct",
            actions=(MoveAction(item.object_id, target.position),),
            rationale="Move directly to the target in one state transition.",
        )
        candidates = [direct]

        horizontal_corner = Position(target.position.x, item.position.y)
        if horizontal_corner not in (item.position, target.position):
            candidates.append(
                CandidatePlan(
                    plan_id="x_then_y",
                    actions=(
                        MoveAction(item.object_id, horizontal_corner),
                        MoveAction(item.object_id, target.position),
                    ),
                    rationale="Reach the target through an x-axis-aligned waypoint.",
                )
            )

        vertical_corner = Position(item.position.x, target.position.y)
        if vertical_corner not in (item.position, target.position):
            candidates.append(
                CandidatePlan(
                    plan_id="y_then_x",
                    actions=(
                        MoveAction(item.object_id, vertical_corner),
                        MoveAction(item.object_id, target.position),
                    ),
                    rationale="Reach the target through a y-axis-aligned waypoint.",
                )
            )
        return tuple(candidates)

