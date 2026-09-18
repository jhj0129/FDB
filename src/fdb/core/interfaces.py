from __future__ import annotations

from typing import Protocol

from .models import CandidateAction, Goal, Observation, Prediction, SafetyDecision, SkillResult


class Environment(Protocol):
    def observe(self) -> Observation: ...
    def execute(self, action: CandidateAction) -> SkillResult: ...


class Skill(Protocol):
    name: str
    capability: str
    provenance: tuple[str, ...]

    def applicable(self, observation: Observation, goal: Goal) -> bool: ...
    def propose(self, observation: Observation, goal: Goal, failure: FailureType | None) -> list[CandidateAction]: ...


class WorldModel(Protocol):
    def predict(self, observation: Observation, candidate: CandidateAction) -> Prediction: ...


class SafetyGate(Protocol):
    def check(self, observation: Observation, prediction: Prediction) -> SafetyDecision: ...


from .models import FailureType  # noqa: E402  (keeps protocol annotations compact)
