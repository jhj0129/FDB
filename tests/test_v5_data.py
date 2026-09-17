from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from fdb.v5.data import FEATURE_NAMES, PushDataset, generate_dataset


def test_generated_dataset_has_reproducible_contract(tmp_path: Path):
    first = generate_dataset(train_count=3, validation_count=2, test_count=2, ood_count=2, seed=7, progress_every=0)
    second = generate_dataset(train_count=3, validation_count=2, test_count=2, ood_count=2, seed=7, progress_every=0)
    assert first.features.shape == (9, len(FEATURE_NAMES))
    assert first.targets.shape == (9, 2)
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.targets, second.targets)
    assert set(first.split) == {"train", "validation", "test", "ood"}
    path = tmp_path / "dataset.npz"
    first.save(path)
    loaded = PushDataset.load(path)
    assert np.array_equal(first.features, loaded.features)
    assert np.array_equal(first.targets, loaded.targets)
