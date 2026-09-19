from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from fdb.memory import EpisodeStore

from .atomic_skills import AtomicSkill, generate_skill_candidates, infer_subgoal
from .models import CandidateAction, FailureType, Goal, Prediction, RuntimeResult


class AtomicSkillWorldModel:
    """Honest mixed predictor: rules for transitions, CNN signal for alignment ranking."""

    def __init__(self, skills: tuple[AtomicSkill, ...], *, use_neural: bool = True) -> None:
        self.skills = {skill.name: skill for skill in skills}
        self.use_neural = use_neural
        self.calls = 0
        self.neural_calls = 0
        self.physics_fallbacks = 0
        self.deterministic_fallbacks = 0

    def predict(self, observation, candidate: CandidateAction) -> Prediction:
        self.calls += 1
        obj = observation.objects[candidate.parameters["object_id"]]
        source = "deterministic_skill_effect"
        probability = 0.92
        uncertainty = 0.08
        if candidate.skill_name in {"align_object", "recover_alignment"} and self.use_neural:
            neural = obj.get("neural_perception") or {}
            if neural.get("fits"):
                self.neural_calls += 1
                source = "shape_fit_cnn+deterministic_skill_effect"
                expected = -float(neural.get("rotation_rad", 0.0))
                proposed = float(candidate.parameters.get("rotation_correction_deg", 0.0))
                error = abs(expected - proposed)
                probability = max(0.05, 1.0 - error / 0.7)
                uncertainty = min(0.8, error / 1.0)
            else:
                self.deterministic_fallbacks += 1
                source = "deterministic_geometry_fallback"
                offset = abs(float(candidate.parameters.get("rotation_correction_deg", 0.0)))
                probability = max(0.1, 0.95 - offset / 180.0)
                uncertainty = 0.35
        effects = dict(self.skills[candidate.skill_name].effects)
        return Prediction(candidate, probability, False, uncertainty, source, effects)


class PhysicalSafetyGate:
    """Final deterministic gate. Learned predictions cannot override these checks."""

    def __init__(self, environment=None) -> None:
        self.environment = environment

    def check(self, observation, prediction: Prediction):
        from .models import SafetyDecision
        import math

        p = prediction.candidate.parameters
        reasons: list[str] = []
        if p.get("object_id") not in observation.objects or p.get("target_id") not in observation.targets:
            reasons.append("INVALID_OBJECT_STATE")
        for key in ("object_pose", "target_pose"):
            pose = p.get(key)
            if pose is None or len(pose) < 3 or any(not math.isfinite(float(v)) for v in pose):
                reasons.append(f"INVALID_{key.upper()}")
                continue
            if not (0.20 <= float(pose[0]) <= 0.75 and -0.38 <= float(pose[1]) <= 0.38):
                reasons.append(f"OUTSIDE_WORKSPACE_{key.upper()}")
        if float(p.get("clearance_m", 0.0)) < 0.08:
            reasons.append("INSUFFICIENT_TABLE_CLEARANCE")
        if prediction.predicted_collision:
            reasons.append("PREDICTED_COLLISION")
        if self.environment is not None:
            reasons.extend(self.environment.preview_safety(prediction.candidate))
        return SafetyDecision(not reasons, tuple(reasons))


class PhysicalAtomicRuntime:
    """State-dependent atomic skill loop with one replay-oriented task episode."""

    def __init__(
        self, environment, skills: tuple[AtomicSkill, ...], world_model: AtomicSkillWorldModel,
        safety_gate: PhysicalSafetyGate, episode_store: EpisodeStore, *, use_memory: bool = True,
    ) -> None:
        self.environment = environment
        self.skills = skills
        self.skill_map = {skill.name: skill for skill in skills}
        self.world_model = world_model
        self.safety_gate = safety_gate
        self.episode_store = episode_store
        self.use_memory = use_memory

    def run(self, goal: Goal, *, max_actions: int = 16) -> RuntimeResult:
        started = perf_counter()
        steps: list[dict[str, Any]] = []
        replans = safety_rejections = memory_hits = 0
        failure: FailureType | None = None
        initial_world_calls = self.world_model.calls
        initial_neural_calls = self.world_model.neural_calls
        initial_fallbacks = self.world_model.deterministic_fallbacks
        initial_observation = self.environment.observe()
        initial_perception_error = self.environment.evaluate_observation(initial_observation)

        for index in range(max_actions):
            observation = self.environment.observe()
            if self._satisfied(observation, goal):
                break
            memories = self.episode_store.search(task_contains=goal.task) if self.use_memory else []
            memory_hits += len(memories)
            subgoal = infer_subgoal(observation, goal)
            candidates, rejected = generate_skill_candidates(self.skills, observation, goal, failure)
            predictions = [self.world_model.predict(observation, candidate) for candidate in candidates]
            checked = [(prediction, self.safety_gate.check(observation, prediction)) for prediction in predictions]
            safe = [prediction for prediction, decision in checked if decision.allowed]
            safety_rejections += len(checked) - len(safe)
            if not safe:
                failure = FailureType.UNSAFE_PLAN
                break
            selected = max(safe, key=lambda item: (item.predicted_success, -item.uncertainty))
            execution = self.environment.execute(selected.candidate)
            after = self.environment.observe()
            actual_effects = {
                key: after.objects[goal.object_id].get(key)
                for key in selected.predicted_next_state
            }
            prediction_error = sum(
                actual_effects[key] != value
                for key, value in selected.predicted_next_state.items()
            )
            steps.append({
                "step": index + 1, "observation": asdict(observation), "current_subgoal": subgoal,
                "candidate_skills": [asdict(item) for item in candidates],
                "rejected_skills": rejected, "predictions": [asdict(item) for item in predictions],
                "safety": [asdict(decision) for _, decision in checked],
                "selected_skill": selected.candidate.skill_name,
                "selection_reason": "safe candidate with highest predicted success and lowest uncertainty",
                "execution": asdict(execution), "actual_next_state": asdict(after),
                "perception_error": self.environment.evaluate_observation(after),
                "prediction_comparison": {
                    "predicted_effects": selected.predicted_next_state,
                    "actual_effects": actual_effects, "effect_mismatch_count": prediction_error,
                    "predicted_success": selected.predicted_success,
                    "actual_success": execution.success,
                    "predicted_collision": selected.predicted_collision,
                    "actual_collision": (
                        bool(execution.metrics.get("robot_table_contact_steps", 0))
                        if "robot_table_contact_steps" in execution.metrics else None
                    ),
                },
            })
            if execution.success:
                failure = None
            else:
                failure = execution.failure_type or FailureType.UNKNOWN
                replans += 1

        final = self.environment.observe()
        agent_success = self._satisfied(final, goal)
        success = agent_success
        if hasattr(self.environment, "evaluator_satisfied"):
            success = agent_success and self.environment.evaluator_satisfied(goal.object_id, goal.target_id)
        if not success and failure is None:
            failure = FailureType.TIMEOUT
        task_episode = {
            "stage": "fdb_physical_runtime_v2", "task": goal.task,
            "environment": {
                "scene_id": initial_observation.scene_id,
                "seed": self.environment.seed,
                "observation_mode": getattr(self.environment, "observation_mode", "unknown"),
                "reset_count": self.environment.reset_count,
                "initial_world_configuration": asdict(self.environment.configuration),
                "headless": True,
            },
            "observation": asdict(initial_observation), "goal": asdict(goal),
            "initial_perception_error": initial_perception_error,
            "candidate_plans": [],
            "decision": {
                "selected_plan_id": "state_dependent_atomic_composition",
                "reason": "each action selected after re-observation",
                "predicted_evaluation": {"success": success},
            },
            "execution": {"decision_steps": steps},
            "result": {
                "final_state": asdict(final),
                "objective_evaluation": {
                    "success": success, "agent_reported_success": agent_success,
                    "steps": len(steps), "replans": replans,
                    "ground_truth": self.environment.evaluator_ground_truth(),
                },
                "user_evaluation": None,
            },
            "reflection": {
                "success_factors": ["persistent_world", "reobservation"] if success else [],
                "failure_causes": [failure.value] if failure else [],
                "learned": "no online learning; trace retained for future adaptation",
                "next_experiment": None, "confidence": 1.0 if success else 0.0,
                "unresolved_questions": [],
            },
            "sources": ["src/fdb/core/physical_scene.py", "src/fdb/core/atomic_skills.py"],
            "skills_used": [step["selected_skill"] for step in steps],
            "model_versions": {"perception": "shape_fit_cnn.msgpack", "controller": "Panda IK+PD"},
            "decision_steps": steps,
            "compute_devices": {
                "perception": "cpu", "world_model": "cpu", "physics": "cpu",
                "controller": "cpu", "runtime": "cpu",
            },
        }
        episode_path = self.episode_store.write(task_episode)
        metrics = {
            "total_task_latency_s": perf_counter() - started, "actions": len(steps),
            "replans": replans, "safety_rejections": safety_rejections,
            "memory_retrieval_count": memory_hits,
            "world_model_calls": self.world_model.calls - initial_world_calls,
            "neural_world_model_calls": self.world_model.neural_calls - initial_neural_calls,
            "deterministic_fallback_calls": self.world_model.deterministic_fallbacks - initial_fallbacks,
            "physics_fallback_calls": 0,
            "final_failure_type": failure.value if failure and not success else None,
        }
        return RuntimeResult(success, len(steps), replans, (str(episode_path),), final, metrics)

    @staticmethod
    def _satisfied(observation, goal: Goal) -> bool:
        obj = observation.objects.get(goal.object_id, {})
        return bool(
            (not goal.success_conditions.get("inside") or obj.get("inside_target") == goal.target_id)
            and (not goal.success_conditions.get("released") or obj.get("released"))
            and (not goal.success_conditions.get("retreated") or obj.get("retreated"))
        )
