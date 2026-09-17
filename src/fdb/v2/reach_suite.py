from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

from .reach import PandaReachExperiment, ReachResult


WORKSPACE_TARGETS = (
    (0.45, 0.15, 0.55),
    (0.45, -0.15, 0.55),
    (0.50, 0.20, 0.65),
    (0.50, -0.20, 0.65),
    (0.60, 0.10, 0.50),
    (0.60, -0.10, 0.50),
    (0.35, 0.15, 0.70),
    (0.35, -0.15, 0.70),
    (0.50, 0.00, 0.75),
    (0.35, 0.00, 0.45),
    (0.65, 0.20, 0.60),
    (0.65, -0.20, 0.60),
)


@dataclass(frozen=True)
class ReachTrial:
    target: tuple[float, float, float]
    selected: ReachResult


@dataclass(frozen=True)
class ReachSuiteResult:
    trials: tuple[ReachTrial, ...]
    success_count: int
    trial_count: int
    success_rate: float
    mean_execution_error_m: float
    max_execution_error_m: float
    joint_limit_violations: int
    contact_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_reach_suite(
    targets: tuple[tuple[float, float, float], ...] = WORKSPACE_TARGETS,
) -> ReachSuiteResult:
    trials = tuple(
        ReachTrial(target, PandaReachExperiment(target).run()[0]) for target in targets
    )
    successes = sum(
        trial.selected.converged
        and trial.selected.execution_error_m <= 0.02
        and trial.selected.joint_limit_violations == 0
        for trial in trials
    )
    errors = [trial.selected.execution_error_m for trial in trials]
    return ReachSuiteResult(
        trials=trials,
        success_count=successes,
        trial_count=len(trials),
        success_rate=successes / len(trials),
        mean_execution_error_m=mean(errors),
        max_execution_error_m=max(errors),
        joint_limit_violations=sum(
            trial.selected.joint_limit_violations for trial in trials
        ),
        contact_count=sum(trial.selected.contact_count for trial in trials),
    )
