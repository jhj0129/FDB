from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

from .pick_place import PandaPickPlaceExperiment, PickPlaceResult, PickPlaceScenario


SCENARIOS = (
    PickPlaceScenario("nominal"),
    PickPlaceScenario("small_light", (0.43, 0.13), (0.56, -0.10), (0.019, 0.019, 0.022), 220, 1.7),
    PickPlaceScenario("small_heavy", (0.47, 0.12), (0.60, -0.08), (0.020, 0.020, 0.023), 480, 2.3),
    PickPlaceScenario("wide", (0.42, 0.10), (0.57, -0.15), (0.026, 0.021, 0.025), 320, 2.0),
    PickPlaceScenario("tall", (0.48, 0.10), (0.55, -0.16), (0.021, 0.021, 0.030), 280, 1.8),
    PickPlaceScenario("low_friction", (0.44, 0.16), (0.60, -0.14), (0.022, 0.022, 0.025), 300, 1.4),
)


@dataclass(frozen=True)
class ManipulationTrial:
    scenario: PickPlaceScenario
    selected: PickPlaceResult


@dataclass(frozen=True)
class ManipulationSuiteResult:
    trials: tuple[ManipulationTrial, ...]
    success_count: int
    trial_count: int
    success_rate: float
    mean_final_xy_error_m: float
    max_final_xy_error_m: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_manipulation_suite() -> ManipulationSuiteResult:
    trials = tuple(
        ManipulationTrial(scenario, PandaPickPlaceExperiment(scenario).run()[0])
        for scenario in SCENARIOS
    )
    successes = sum(trial.selected.success for trial in trials)
    errors = [trial.selected.final_xy_error_m for trial in trials]
    return ManipulationSuiteResult(
        trials=trials,
        success_count=successes,
        trial_count=len(trials),
        success_rate=successes / len(trials),
        mean_final_xy_error_m=mean(errors),
        max_final_xy_error_m=max(errors),
    )
