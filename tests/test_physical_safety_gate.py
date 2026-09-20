from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from fdb.core.models import CandidateAction, Observation, Prediction
from fdb.core.physical_runtime import PhysicalSafetyGate
from fdb.core.physical_scene import PhysicalPersistentShapeEnvironment
from fdb.v2.pose_reach import PandaPoseReachExperiment


def candidate(**changes):
    parameters = {"object_id": "triangle_01", "target_id": "triangle_target",
                  "object_pose": [0.4, 0.1, 0.33], "target_pose": [0.6, 0.1, 0.29],
                  "clearance_m": 0.1}
    parameters.update(changes)
    return CandidateAction("safety-probe", "reach_object", parameters)


@pytest.mark.parametrize("changes,collision,reason", [
    ({"object_pose": [0.9, 0.1, 0.33]}, False, "OUTSIDE_WORKSPACE_OBJECT_POSE"),
    ({"target_pose": [0.6, float("nan"), 0.29]}, False, "INVALID_TARGET_POSE"),
    ({"clearance_m": 0.01}, False, "INSUFFICIENT_TABLE_CLEARANCE"),
    ({}, True, "PREDICTED_COLLISION"),
])
def test_physical_gate_rejects_unsafe_predictions(changes, collision, reason):
    observation = Observation("test", {"triangle_01": {}}, {"triangle_target": {}})
    prediction = Prediction(candidate(**changes), 1.0, collision, 0.0, "test")
    decision = PhysicalSafetyGate().check(observation, prediction)
    assert not decision.allowed
    assert reason in decision.reasons


@pytest.mark.parametrize("target,limit,reason", [
    (3.0, 2.0, "JOINT_LIMIT"),
    (5.0, 10.0, "VELOCITY_LIMIT"),
    (14.0, 20.0, "ACCELERATION_LIMIT"),
    (float("nan"), 2.0, "INVALID_TRAJECTORY"),
    (None, 2.0, "IK_FAILURE"),
])
def test_physical_preview_rejects_unsafe_ik(monkeypatch, target, limit, reason):
    environment = object.__new__(PhysicalPersistentShapeEnvironment)
    environment.model = SimpleNamespace(jnt_range=np.tile([-limit, limit], (7, 1)))
    environment.data = SimpleNamespace(ctrl=np.zeros(7))

    def solve(self):
        if target is None:
            raise RuntimeError("injected IK failure")
        return (SimpleNamespace(target_qpos=np.full(7, target)),)

    monkeypatch.setattr(PandaPoseReachExperiment, "run", solve)
    assert reason in environment.preview_safety(candidate())
