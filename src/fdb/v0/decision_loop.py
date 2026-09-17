from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from fdb.memory import EpisodeStore

from .interpreter import TaskInterpreter
from .models import CandidatePlan, Evaluation, WorldState
from .planner import CandidatePlanner
from .simulator import SymbolicSimulator


@dataclass(frozen=True)
class DecisionResult:
    final_state: WorldState
    selected_plan: CandidatePlan
    evaluation: Evaluation
    episode_path: Path | None


class DecisionLoop:
    """Task -> goal -> candidates -> simulation -> selection -> action -> evaluation -> memory."""

    def __init__(self, episode_store: EpisodeStore | None = None) -> None:
        self.interpreter = TaskInterpreter()
        self.planner = CandidatePlanner()
        self.simulator = SymbolicSimulator()
        self.episode_store = episode_store

    def run(self, task: str, initial_state: WorldState) -> DecisionResult:
        goal = self.interpreter.interpret(task)
        candidates = self.planner.create_candidates(initial_state, goal)
        simulated = [
            (plan, *self.simulator.run(initial_state, plan, goal)) for plan in candidates
        ]
        selected_plan, _, predicted_evaluation = max(
            simulated,
            key=lambda item: (item[2].score, -item[2].action_count, item[0].plan_id),
        )

        # Execution is intentionally a separate pass from internal simulation.
        final_state, actual_evaluation = self.simulator.run(initial_state, selected_plan, goal)
        episode_path = None
        if self.episode_store:
            episode_path = self.episode_store.write(
                {
                    "stage": "v0",
                    "task": task,
                    "observation": initial_state.to_dict(),
                    "goal": asdict(goal),
                    "candidate_plans": [
                        {
                            "plan": asdict(plan),
                            "predicted_evaluation": asdict(evaluation),
                        }
                        for plan, _, evaluation in simulated
                    ],
                    "decision": {
                        "selected_plan_id": selected_plan.plan_id,
                        "reason": "Highest objective simulation score with fewer actions as tie-breaker.",
                        "predicted_evaluation": asdict(predicted_evaluation),
                    },
                    "execution": {"actions": [asdict(action) for action in selected_plan.actions]},
                    "result": {
                        "final_state": final_state.to_dict(),
                        "objective_evaluation": asdict(actual_evaluation),
                        "user_evaluation": None,
                    },
                    "reflection": {
                        "success_factors": ["Candidate was validated before execution"]
                        if actual_evaluation.success
                        else [],
                        "failure_causes": list(actual_evaluation.violations),
                        "learned": "v0 records evidence but does not auto-promote a skill.",
                        "next_experiment": "Add obstacles and collision-aware planning in v1.",
                        "confidence": 1.0 if actual_evaluation.success else 0.2,
                        "unresolved_questions": [],
                    },
                    "sources": [],
                    "skills_used": ["v0.object_relocation"],
                }
            )
        return DecisionResult(final_state, selected_plan, actual_evaluation, episode_path)

