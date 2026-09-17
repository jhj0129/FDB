import pytest
from dataclasses import replace

pytest.importorskip("mujoco")

from fdb.challenge.multi_object_sorting import (
    OBJECTS, classify_object, classify_object_neural, run_sorting,
)


def test_observable_attributes_map_to_three_sorting_classes():
    assert [classify_object(obj) for obj in OBJECTS] == [
        "red_small", "green_wide", "blue_tall"
    ]


def test_portable_neural_ensemble_classifies_scene_objects():
    predictions = [classify_object_neural(obj) for obj in OBJECTS]
    assert [label for label, _ in predictions] == ["red_small", "green_wide", "blue_tall"]
    assert min(confidence for _, confidence in predictions) >= 0.95


def test_portable_neural_ensemble_rejects_ambiguous_ood_object():
    ambiguous = replace(
        OBJECTS[0], color="yellow", size_class="unknown",
        rgba=(0.50, 0.50, 0.05, 1.0), half_size=(0.030, 0.012, 0.015),
    )
    label, _ = classify_object_neural(ambiguous)
    assert label == "unknown"


def test_three_objects_are_physically_sorted_without_robot_table_contact():
    metrics = run_sorting()
    assert metrics["classification_accuracy"] == 1.0
    assert metrics["classification_method"] == "five_member_mlp_ensemble"
    assert metrics["sorting_success_count"] == 3
    assert metrics["hard_safety_pass"] is True
