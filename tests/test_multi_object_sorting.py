import pytest

pytest.importorskip("mujoco")

from fdb.challenge.multi_object_sorting import OBJECTS, classify_object, run_sorting


def test_observable_attributes_map_to_three_sorting_classes():
    assert [classify_object(obj) for obj in OBJECTS] == [
        "red_small", "green_wide", "blue_tall"
    ]


def test_three_objects_are_physically_sorted_without_robot_table_contact():
    metrics = run_sorting()
    assert metrics["classification_accuracy"] == 1.0
    assert metrics["sorting_success_count"] == 3
    assert metrics["hard_safety_pass"] is True
