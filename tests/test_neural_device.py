from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fdb.v5.network import NeuralDynamicsEnsemble


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_same_checkpoint_cpu_cuda_predictions_match():
    path = Path("models/v5/push_dynamics_ensemble.pt")
    cpu = NeuralDynamicsEnsemble.load(path)
    gpu = NeuralDynamicsEnsemble.load(path, device="cuda")
    assert all(next(model.parameters()).device.type == "cuda" for model in gpu.models)
    rng = np.random.default_rng(0)
    for _ in range(10):
        vector = rng.uniform(cpu.feature_min, cpu.feature_max).astype(np.float32)
        expected, actual = cpu.predict(vector), gpu.predict(vector)
        np.testing.assert_allclose(expected.final_position, actual.final_position, rtol=1e-4, atol=1e-5)
        assert expected.in_distribution == actual.in_distribution
        assert expected.confident == actual.confident
