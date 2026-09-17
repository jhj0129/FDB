from __future__ import annotations

from .models import CandidatePlan, Evaluation, Goal, WorldState


class SymbolicSimulator:
    """Side-effect-free internal simulation for v0 candidate plans."""

    def run(self, initial: WorldState, plan: CandidatePlan, goal: Goal) -> tuple[WorldState, Evaluation]:
        state = initial
        path_length = 0
        violations: list[str] = []
        for action in plan.actions:
            before = next(item for item in state.objects if item.object_id == action.object_id)
            try:
                state.validate_position(action.destination)
                path_length += before.position.distance_to(action.destination)
                state = state.move(action)
            except ValueError as error:
                violations.append(str(error))
                break

        item = state.object_by_color(goal.object_color)
        target = state.zone_by_color(goal.target_color)
        success = item.position == target.position and not violations
        # Objective score is deterministic and intentionally separate from a future learned critic.
        score = (100.0 if success else 0.0) - path_length - len(plan.actions) - 25 * len(violations)
        return state, Evaluation(
            success=success,
            score=score,
            path_length=path_length,
            action_count=len(plan.actions),
            violations=tuple(violations),
        )

