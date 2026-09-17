from __future__ import annotations

from .models import PushCandidate


class PushCandidatePlanner:
    """Generate a small, explicit force/duration search set."""

    def create_candidates(self) -> tuple[PushCandidate, ...]:
        return (
            PushCandidate(
                plan_id="gentle_long",
                force_newtons=2.0,
                force_steps=160,
                settle_steps=500,
                rationale="Low force applied for longer to limit overshoot.",
            ),
            PushCandidate(
                plan_id="balanced",
                force_newtons=4.0,
                force_steps=120,
                settle_steps=500,
                rationale="Moderate force and duration for a balanced transfer.",
            ),
            PushCandidate(
                plan_id="firm_short",
                force_newtons=7.0,
                force_steps=80,
                settle_steps=500,
                rationale="Higher force applied briefly to overcome static friction.",
            ),
        )
