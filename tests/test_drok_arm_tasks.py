from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from fdb.challenge.drok_arm_sorting import (
    DROK_OBJECTS,
    HARD_DROK_OBJECTS,
    HARD_YAWS,
    run_sorting,
)
from fdb.challenge.drok_arm_tasks import DEFAULT_DROK_ROOT, analyze_structure, run_pick_place


pytestmark = pytest.mark.skipif(
    not (DEFAULT_DROK_ROOT / "src/drok_arm_mujoco/model/drok_arm.xml").exists(),
    reason="외부 DROK_ARM_Sim_only 저장소가 필요합니다.",
)


def test_drok_structure_exposes_model_contract_mismatches() -> None:
    structure = analyze_structure()
    assert structure.arm_joints == 6
    assert structure.gripper_joints == 2
    assert structure.base_z_documented_m == 1.0
    assert structure.base_z_model_m == 0.0
    assert structure.home_floor_penetration_m > 0.0
    assert structure.finger_self_collision_position_m < 0.01


def test_drok_pick_place_lifts_places_and_avoids_table() -> None:
    result = run_pick_place()
    assert result.success is True
    assert result.lift_height_m >= 0.08
    assert result.final_xy_error_m <= 0.025
    assert result.robot_table_contact_steps == 0
    assert result.visual_contact_steps > 0
    assert -0.001 <= result.closest_two_sided_visual_gap_m <= 0.0
    assert result.minimum_visible_gripper_table_clearance_m >= 0.0


def test_drok_neural_sorting_completes_three_objects_safely() -> None:
    metrics = run_sorting(objects=DROK_OBJECTS)
    assert metrics["classification_accuracy"] == 1.0
    assert metrics["sorting_success_count"] == len(DROK_OBJECTS)
    assert metrics["hard_safety_pass"] is True
    assert all(result["visible_contact_verified"] for result in metrics["results"])


def test_drok_rotated_objects_clear_central_obstacle() -> None:
    metrics = run_sorting(
        objects=HARD_DROK_OBJECTS,
        object_yaws=HARD_YAWS,
        add_obstacle=True,
    )
    assert metrics["sorting_success_count"] == len(HARD_DROK_OBJECTS)
    assert metrics["obstacle_contact_steps"] == 0
    assert metrics["minimum_visible_gripper_table_clearance_m"] >= 0.0
    assert all(
        -0.001 <= result["closest_two_sided_visual_gap_m"] <= 0.0
        for result in metrics["results"]
    )
