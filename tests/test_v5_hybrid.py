from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from fdb.v1.models import PhysicsScenario
from fdb.v1.planner import PushCandidatePlanner
from fdb.v5.hybrid import SafeHybridPushPlanner


MODEL = Path("models/v5/push_dynamics_ensemble.npz")


def test_neural_screening_is_physically_verified():
    result = SafeHybridPushPlanner(MODEL).decide(PushCandidatePlanner().create_candidates())
    assert result.selected.source == "neural_ensemble"
    assert result.verified_execution.evaluation.success is True
    assert result.neural_to_physics_error_m is not None
    assert result.neural_to_physics_error_m <= 0.03


def test_out_of_distribution_inputs_force_physics_fallback():
    scenario = PhysicsScenario(
        "ood_test", mass_scale=2.3, sliding_friction=1.75,
        initial_offset_x=0.22, initial_offset_y=-0.2,
    )
    result = SafeHybridPushPlanner(MODEL).decide(
        PushCandidatePlanner().create_candidates(), scenario
    )
    assert all(candidate.source == "physics_fallback" for candidate in result.candidates)
    assert all("학습 분포 밖 입력" in candidate.fallback_reasons for candidate in result.candidates)
