import numpy as np

from fdb.challenge.gait_skill_learning import time_features


def test_time_features_are_deterministic_and_cover_multiple_frequencies():
    features = time_features(101, harmonics=4)
    assert features.shape == (101, 9)
    assert np.allclose(features[0, 1::2], 0.0, atol=1e-6)
    assert np.allclose(features[0, 2::2], 1.0, atol=1e-6)
    assert np.isclose(features[0, 0], 0.0)
    assert np.isclose(features[-1, 0], 1.0)
