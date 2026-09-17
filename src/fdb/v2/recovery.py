from __future__ import annotations

from dataclasses import dataclass

from .grasp_lift import GraspLiftResult, PandaGraspLiftExperiment


@dataclass(frozen=True)
class FailureDiagnosis:
    failure_type: str
    evidence: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class RecoveryResult:
    initial_attempt: GraspLiftResult
    diagnosis: FailureDiagnosis
    recovery_attempt: GraspLiftResult
    recovered: bool
    reset_based: bool = True


class ManipulationFailureDetector:
    def diagnose_grasp_lift(self, result: GraspLiftResult) -> FailureDiagnosis:
        evidence: list[str] = []
        if result.lift_height_m < 0.08:
            evidence.append(f"lift_height_m={result.lift_height_m:.6f}<0.08")
        if result.final_object_to_hand_distance_m > 0.16:
            evidence.append(
                f"retention_distance_m={result.final_object_to_hand_distance_m:.6f}>0.16"
            )
        if result.forbidden_contact_steps:
            evidence.append(
                f"forbidden_contact_steps={result.forbidden_contact_steps}>0"
            )
        if not result.stable:
            return FailureDiagnosis("simulation_instability", tuple(evidence), 1.0)
        if result.lift_height_m < 0.08 and result.final_object_to_hand_distance_m > 0.16:
            return FailureDiagnosis("missed_grasp", tuple(evidence), 0.95)
        if result.forbidden_contact_steps:
            return FailureDiagnosis("object_recontact_or_drop", tuple(evidence), 0.9)
        if result.success:
            return FailureDiagnosis("none", (), 1.0)
        return FailureDiagnosis("unclassified_manipulation_failure", tuple(evidence), 0.5)


class ResetBasedRecoveryExperiment:
    """Diagnose a failed sandbox rollout and retry a validated alternative from reset."""

    def run(self) -> RecoveryResult:
        experiment = PandaGraspLiftExperiment()
        candidates = experiment.candidates()
        initial = experiment.run_candidate(candidates[2])  # deliberately high and missed
        diagnosis = ManipulationFailureDetector().diagnose_grasp_lift(initial)
        if diagnosis.failure_type != "missed_grasp":
            return RecoveryResult(initial, diagnosis, initial, False)
        recovery = experiment.run_candidate(candidates[0])  # evidence-backed low grasp
        return RecoveryResult(initial, diagnosis, recovery, recovery.success)

