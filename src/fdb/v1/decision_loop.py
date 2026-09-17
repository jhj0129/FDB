from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from fdb.memory import EpisodeStore

from .environment import MujocoPushEnvironment
from .models import PhysicsRollout
from .planner import PushCandidatePlanner


@dataclass(frozen=True)
class PhysicsDecisionResult:
    selected_prediction: PhysicsRollout
    execution: PhysicsRollout
    episode_path: Path | None


class PhysicsDecisionLoop:
    """Plan, simulate candidates, select, independently execute, evaluate, remember."""

    def __init__(self, episode_store: EpisodeStore | None = None) -> None:
        self.planner = PushCandidatePlanner()
        self.environment = MujocoPushEnvironment()
        self.episode_store = episode_store

    def run(self, task: str = "Move the red object to the blue target") -> PhysicsDecisionResult:
        candidates = self.planner.create_candidates()
        predictions = tuple(self.environment.simulate(plan) for plan in candidates)
        selected = max(
            predictions,
            key=lambda rollout: (
                rollout.evaluation.success,
                rollout.evaluation.score,
                -rollout.evaluation.final_distance,
            ),
        )
        execution = self.environment.simulate(selected.plan)

        episode_path = None
        if self.episode_store:
            episode_path = self.episode_store.write(
                {
                    "stage": "v1",
                    "task": task,
                    "observation": {
                        "environment": "MuJoCo",
                        "model": "fdb.v1/assets/push.xml",
                        "initial_position": selected.initial_position,
                        "target_position": selected.target_position,
                    },
                    "goal": {
                        "relation": "puck center within target radius",
                        "success_radius_m": self.environment.success_radius,
                    },
                    "candidate_plans": [asdict(item) for item in predictions],
                    "decision": {
                        "selected_plan_id": selected.plan.plan_id,
                        "reason": "Completion first, then highest objective physics score.",
                        "predicted_evaluation": asdict(selected.evaluation),
                    },
                    "execution": {"plan": asdict(selected.plan)},
                    "result": {
                        "final_state": {"puck_position": execution.final_position},
                        "objective_evaluation": asdict(execution.evaluation),
                        "user_evaluation": None,
                    },
                    "reflection": {
                        "success_factors": ["Selected plan succeeded in a fresh physics rollout"]
                        if execution.evaluation.success
                        else [],
                        "failure_causes": []
                        if execution.evaluation.success
                        else ["No candidate stopped inside the target radius"],
                        "learned": "Open-loop force candidates can be compared before execution.",
                        "next_experiment": "Add feedback control and randomized mass/friction trials.",
                        "confidence": 0.8 if execution.evaluation.success else 0.3,
                        "unresolved_questions": [
                            "How robust is the selected plan to mass and friction variation?"
                        ],
                    },
                    "sources": [
                        "research/research_20260917_mujoco_python_api.json"
                    ],
                    "skills_used": ["v1.physics_push"],
                }
            )
        return PhysicsDecisionResult(selected, execution, episode_path)

