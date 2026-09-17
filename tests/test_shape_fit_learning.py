import numpy as np

from fdb.challenge.shape_fit_learning import make_dataset


def test_visual_dataset_contains_known_unknown_fit_and_nonfit_examples():
    images, labels = make_dataset(500, seed=9)
    assert images.shape == (500, 32, 64, 3)
    assert 0.0 <= float(images.min()) < float(images.max()) <= 1.0
    assert set(labels["object_shape"]) == {0, 1, 2, 3, 4}
    assert set(labels["hole_shape"]) == {0, 1, 2, 3, 4}
    assert {False, True} == set(labels["fits"].astype(bool))
    assert np.all(labels["fits"][(labels["object_shape"] == 4) | (labels["hole_shape"] == 4)] == 0)


def test_fit_label_requires_matching_shape_and_clearance():
    _, labels = make_dataset(1000, seed=10)
    positives = labels["fits"] > 0.5
    assert np.all(labels["object_shape"][positives] == labels["hole_shape"][positives])
    assert np.all(labels["object_shape"][positives] < 4)
