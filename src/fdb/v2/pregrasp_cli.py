from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fdb.memory import EpisodeStore

from .pregrasp import CollisionAwarePregraspExperiment


def main() -> None:
    experiment = CollisionAwarePregraspExperiment()
    selected, candidates = experiment.run()
    episode = EpisodeStore(Path("episodes")).write(
        {
            "stage": "v2",
            "task": "Select a collision-free Panda pre-grasp pose around an obstacle.",
            "observation": {"obstacle_position": experiment.obstacle_position, "obstacle_radius": experiment.obstacle_radius},
            "goal": {"contact_step_count": 0, "position_error_m_max": 0.02},
            "candidate_plans": [asdict(item) for item in candidates],
            "decision": {
                "selected_plan_id": selected.plan.plan_id,
                "reason": "Collision-free validity is mandatory before objective score.",
                "predicted_evaluation": asdict(selected),
            },
            "execution": {"target": selected.plan.target},
            "result": {"final_state": {}, "objective_evaluation": asdict(selected), "user_evaluation": None},
            "reflection": {
                "success_factors": ["Rejected accurate but colliding candidates before score comparison."],
                "failure_causes": ["Direct center and centered-high candidates contacted the obstacle."],
                "learned": "Safety feasibility must be lexicographically prior to endpoint quality.",
                "next_experiment": "Approach from the selected pre-grasp and close the gripper on a movable object.",
                "confidence": 0.85,
                "unresolved_questions": ["Which contacts should be allowed during grasp closure?"],
            },
            "sources": ["research/research_20260917_panda_model.json"],
            "skills_used": ["v2.collision_aware_pregrasp"],
        }
    )
    print(json.dumps({"selected": asdict(selected), "candidates": [asdict(item) for item in candidates], "episode": str(episode)}, indent=2))


if __name__ == "__main__":
    main()
