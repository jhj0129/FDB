import pytest

pytest.importorskip("mujoco")

from fdb.challenge.sorting_robustness import SCENARIOS
from fdb.challenge.multi_object_sorting import run_sorting


def test_jittered_sources_and_targets_remain_safe_and_sorted():
    _, objects = SCENARIOS[1]
    metrics = run_sorting(objects=objects)
    assert metrics["sorting_success_count"] == 3
    assert metrics["hard_safety_pass"] is True
