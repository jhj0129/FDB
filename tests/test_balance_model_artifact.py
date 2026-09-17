import json
from pathlib import Path

import numpy as np


def test_balance_model_is_portable_and_conservative():
    with np.load(Path("models/v5/humanoid_balance_ensemble.npz")) as model:
        assert int(model["ensemble_size"][0]) == 5
        assert len(model["feature_names"]) == 12
        assert np.isclose(float(model["stable_probability_threshold"][0]), 0.9, atol=1e-6)
    metrics = json.loads(Path("experiments/0018_humanoid_balance.metrics.json").read_text())
    assert metrics["splits"]["test"]["false_stable_count"] == 0
    assert metrics["splits"]["test"]["stable_precision"] == 1.0
    assert set(metrics["test_by_robot"]) == {
        "unitree_g1", "booster_t1", "robotis_op3", "berkeley_humanoid"
    }


def test_balance_dataset_contains_both_outcomes_and_all_robots():
    with np.load(Path("data/v5/humanoid_balance.npz")) as dataset:
        assert len(dataset["stable"]) == 192
        assert 80 <= int(dataset["stable"].sum()) <= 112
        assert len(np.unique(dataset["robot"])) == 4
