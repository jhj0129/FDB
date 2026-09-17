from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fdb.memory import EpisodeStore

from .grasp_lift import PandaGraspLiftExperiment


def main() -> None:
    selected, candidates = PandaGraspLiftExperiment().run()
    episode = EpisodeStore(Path("episodes")).write(
        {
            "stage": "v2",
            "task": "Approach, grasp, and lift a box with the Panda gripper.",
            "observation": {"object_position": PandaGraspLiftExperiment.object_position},
            "goal": {"lift_height_m_min": 0.08, "retain_object": True},
            "candidate_plans": [asdict(item) for item in candidates],
            "decision": {"selected_plan_id": selected.plan.plan_id, "reason": "Task success first, then lift and retention score.", "predicted_evaluation": asdict(selected)},
            "execution": {"sequence": ["pregrasp", "descend", "close", "lift"]},
            "result": {"final_state": {"object_z": selected.final_object_z}, "objective_evaluation": asdict(selected), "user_evaluation": None},
            "reflection": {
                "success_factors": ["Compared grasp-height candidates and preserved the free object's initial pose."],
                "failure_causes": ["Initial sphere slipped; unextended keyframe reset misplaced the free object."],
                "learned": "Added free joints require explicit qpos initialization when reusing an older keyframe.",
                "next_experiment": "Place the lifted object into a target region and release it.",
                "confidence": 0.75,
                "unresolved_questions": ["How robust is grasp to object size, mass, and pose?"],
            },
            "sources": ["research/research_20260917_panda_model.json"],
            "skills_used": ["v2.panda_grasp_lift"],
        }
    )
    print(json.dumps({"selected": asdict(selected), "candidates": [asdict(item) for item in candidates], "episode": str(episode)}, indent=2))


if __name__ == "__main__":
    main()
