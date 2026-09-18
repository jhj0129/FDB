from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from fdb.memory import EpisodeStore

from .models import FailureType, Goal, RuntimeResult


class FDBRuntime:
    """Goal-directed observe/predict/safety/act/evaluate/replan loop."""

    def __init__(self, environment, skills, world_model, safety_gate, episode_store: EpisodeStore) -> None:
        self.environment = environment
        self.skills = skills
        self.world_model = world_model
        self.safety_gate = safety_gate
        self.episode_store = episode_store

    def run(self, goal: Goal, *, max_actions: int = 6) -> RuntimeResult:
        started = perf_counter()
        initial_world_calls = self.world_model.calls
        initial_physics_fallbacks = self.world_model.physics_fallbacks
        initial_deterministic_fallbacks = getattr(self.world_model, "deterministic_fallbacks", 0)
        episodes: list[str] = []
        failure: FailureType | None = None
        replans = 0
        counters = {"memory_retrieval_count": 0, "safety_rejections": 0}
        timings = {"observation_s": 0.0, "memory_retrieval_s": 0.0,
                   "planning_s": 0.0, "prediction_s": 0.0, "execution_s": 0.0}

        for action_index in range(max_actions):
            tick = perf_counter()
            observation = self.environment.observe()
            timings["observation_s"] += perf_counter() - tick
            if self._satisfied(observation, goal):
                episodes.append(str(self._write_terminal_episode(
                    goal, observation, True, None, "goal already satisfied",
                )))
                break

            tick = perf_counter()
            memories = self.episode_store.search(task_contains=goal.task)
            counters["memory_retrieval_count"] += len(memories)
            timings["memory_retrieval_s"] += perf_counter() - tick

            tick = perf_counter()
            available = self.skills.available(observation, goal)
            candidates = [candidate for skill in available for candidate in skill.propose(observation, goal, failure)]
            timings["planning_s"] += perf_counter() - tick
            if not candidates:
                failure = FailureType.TARGET_NOT_FOUND
                episodes.append(str(self._write_terminal_episode(
                    goal, observation, False, failure, "no applicable skill candidate",
                )))
                break

            tick = perf_counter()
            predictions = [self.world_model.predict(observation, candidate) for candidate in candidates]
            timings["prediction_s"] += perf_counter() - tick
            checked = [(prediction, self.safety_gate.check(observation, prediction)) for prediction in predictions]
            safe = [prediction for prediction, decision in checked if decision.allowed]
            counters["safety_rejections"] += len(checked) - len(safe)
            if not safe:
                failure = FailureType.UNSAFE_PLAN
                episodes.append(str(self._write_terminal_episode(
                    goal, observation, False, failure, "all candidates rejected by safety gate",
                )))
                break
            selected = max(safe, key=lambda item: (item.predicted_success, -item.uncertainty))

            tick = perf_counter()
            result = self.environment.execute(selected.candidate)
            timings["execution_s"] += perf_counter() - tick
            after = self.environment.observe()
            success = result.success and self._satisfied(after, goal)
            episode = {
                "stage": "fdb_runtime_v1",
                "task": goal.task,
                "environment": {"scene_id": observation.scene_id, "sequence": observation.sequence},
                "observation": asdict(observation),
                "goal": asdict(goal),
                "retrieved_memories": [item.get("episode_id") for item in memories[-5:]],
                "candidate_plans": [asdict(item) for item in candidates],
                "predictions": [asdict(item) for item in predictions],
                "decision": {
                    "selected_plan_id": selected.candidate.action_id,
                    "reason": "안전 게이트를 통과한 후보 중 예측 성공률 최대",
                    "predicted_evaluation": asdict(selected),
                    "selected_action": asdict(selected.candidate),
                },
                "safety_result": [asdict(decision) for _, decision in checked],
                "execution": asdict(result),
                "result": {
                    "final_state": asdict(after),
                    "objective_evaluation": {"success": success, **result.metrics},
                    "user_evaluation": None,
                },
                "reflection": {
                    "success_factors": ["goal_state_verified"] if success else [],
                    "failure_causes": [result.failure_type.value] if result.failure_type else [],
                    "learned": "episode retained for later validated promotion",
                    "next_experiment": "retry alternative candidate" if not success else None,
                    "confidence": selected.predicted_success,
                    "unresolved_questions": [],
                    "failure_type": result.failure_type.value if result.failure_type else None,
                    "retry_required": not success,
                },
                "sources": list(self.skills.get(selected.candidate.skill_name).provenance),
                "skills_used": [selected.candidate.skill_name],
                "model_versions": {"world_model": type(self.world_model).__name__},
            }
            episodes.append(str(self.episode_store.write(episode)))
            if success:
                failure = None
                break
            failure = result.failure_type or FailureType.UNKNOWN
            replans += 1

        final = self.environment.observe()
        success = self._satisfied(final, goal)
        metrics: dict[str, Any] = {
            **counters, **timings,
            "total_task_latency_s": perf_counter() - started,
            "world_model_calls": self.world_model.calls - initial_world_calls,
            "physics_fallback_calls": self.world_model.physics_fallbacks - initial_physics_fallbacks,
            "deterministic_fallback_calls": (
                getattr(self.world_model, "deterministic_fallbacks", 0)
                - initial_deterministic_fallbacks
            ),
            "final_failure_type": None if success or failure is None else failure.value,
        }
        return RuntimeResult(success, len(episodes), replans, tuple(episodes), final, metrics)

    def _write_terminal_episode(self, goal, observation, success, failure, reason) -> Path:
        failure_name = failure.value if failure else None
        return self.episode_store.write({
            "stage": "fdb_runtime_v1", "task": goal.task,
            "environment": {"scene_id": observation.scene_id, "sequence": observation.sequence},
            "observation": asdict(observation), "goal": asdict(goal),
            "retrieved_memories": [], "candidate_plans": [],
            "decision": {
                "selected_plan_id": "none", "reason": reason,
                "predicted_evaluation": {"success": success},
            },
            "execution": {"performed": False},
            "result": {
                "final_state": asdict(observation),
                "objective_evaluation": {"success": success}, "user_evaluation": None,
            },
            "reflection": {
                "success_factors": ["goal_already_satisfied"] if success else [],
                "failure_causes": [failure_name] if failure_name else [],
                "learned": "terminal runtime outcome retained",
                "next_experiment": None, "confidence": 1.0 if success else 0.0,
                "unresolved_questions": [], "failure_type": failure_name,
                "retry_required": False,
            },
            "sources": [], "skills_used": [], "model_versions": {},
        })

    @staticmethod
    def _satisfied(observation, goal: Goal) -> bool:
        state = observation.objects.get(goal.object_id)
        if state is None:
            return False
        conditions = goal.success_conditions
        return bool(
            (not conditions.get("inside", False) or state.get("inside_target") == goal.target_id)
            and (not conditions.get("released", False) or state.get("released"))
        )
