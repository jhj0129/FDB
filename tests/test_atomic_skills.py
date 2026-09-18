from fdb.core.atomic_skills import build_shape_manipulation_skills, generate_skill_candidates, infer_subgoal
from fdb.core.models import FailureType, Goal, Observation


def state(**changes):
    obj = {
        "shape": "triangle", "visible": True, "ee_near": False, "grasped": False,
        "lifted": False, "aligned": False, "near_target": False, "inserted": False,
        "released": False, "pose": [0.4, 0.1, 0.33], "yaw_error_deg": 30.0,
        "last_failure": None,
    }
    obj.update(changes)
    return Observation("scene", {"triangle_01": obj}, {
        "triangle_target": {"shape": "triangle", "visible": True, "pose": [0.6, 0.1, 0.29]},
    })


GOAL = Goal("object_to_target", "triangle_01", "triangle_target", {"inside": True, "released": True})


def selected_names(observation):
    candidates, _ = generate_skill_candidates(build_shape_manipulation_skills(), observation, GOAL)
    return {candidate.skill_name for candidate in candidates}


def test_preconditions_select_skill_from_state_not_shape_sequence() -> None:
    assert selected_names(state()) == {"reach_object"}
    assert selected_names(state(ee_near=True)) == {"grasp_object"}
    assert selected_names(state(ee_near=True, grasped=True)) == {"lift_object"}
    assert selected_names(state(ee_near=True, grasped=True, lifted=True)) == {"align_object"}
    assert selected_names(state(ee_near=True, grasped=True, lifted=True, aligned=True)) == {"move_to_target"}


def test_alignment_failure_changes_recovery_path_without_reset() -> None:
    observation = state(
        ee_near=True, grasped=True, lifted=True, last_failure=FailureType.ALIGNMENT_FAILURE,
    )
    candidates, rejected = generate_skill_candidates(
        build_shape_manipulation_skills(), observation, GOAL, FailureType.ALIGNMENT_FAILURE,
    )
    assert {item.skill_name for item in candidates} == {"recover_alignment"}
    assert len(candidates) == 3
    assert "align_object" in rejected
    assert infer_subgoal(observation, GOAL) == "recover_alignment"


def test_skill_effects_are_declared() -> None:
    skills = {skill.name: skill for skill in build_shape_manipulation_skills()}
    assert skills["grasp_object"].effects == {"grasped": True}
    assert skills["release_object"].effects == {"released": True, "grasped": False}
    assert FailureType.GRASP_FAILURE in skills["grasp_object"].failures


def test_goal_does_not_hide_other_scene_objects() -> None:
    observation = state()
    objects = dict(observation.objects)
    objects["circle_01"] = {"shape": "circle", "visible": True}
    expanded = Observation(observation.scene_id, objects, observation.targets)
    generate_skill_candidates(build_shape_manipulation_skills(), expanded, GOAL)
    assert set(expanded.objects) == {"triangle_01", "circle_01"}
